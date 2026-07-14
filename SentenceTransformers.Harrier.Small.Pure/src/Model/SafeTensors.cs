using System.Buffers.Binary;
using System.Text.Json;
using SentenceTransformers.Harrier.Small.Pure.Numerics;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// Minimal, pure-managed reader for the Hugging Face <c>safetensors</c> format.
///
/// Layout: an 8-byte little-endian unsigned integer giving the length of a JSON header, followed by
/// that JSON header, followed by the raw tensor bytes. Each header entry maps a tensor name to its
/// <c>dtype</c>, <c>shape</c> and <c>data_offsets</c> (a [begin, end) byte range into the data
/// section). See https://github.com/huggingface/safetensors for the specification.
///
/// Only the dtypes the Harrier weights actually use are supported: BF16, F16 and F32.
/// </summary>
internal sealed class SafeTensors
{
    private readonly byte[] _bytes;         // the whole file
    private readonly long _dataStart;       // byte offset where the tensor data section begins
    private readonly Dictionary<string, Entry> _entries;

    private readonly record struct Entry(string Dtype, int[] Shape, long Begin, long End);

    private SafeTensors(byte[] bytes, long dataStart, Dictionary<string, Entry> entries)
    {
        _bytes = bytes;
        _dataStart = dataStart;
        _entries = entries;
    }

    public IReadOnlyCollection<string> Names => _entries.Keys;

    public int[] Shape(string name) => GetEntry(name).Shape;

    public bool Contains(string name) => _entries.ContainsKey(name);

    /// <summary>Reads the whole file into memory and parses the header. The weights file is a few
    /// hundred MB, so we map it once and keep the raw bytes around for lazy per-tensor widening.</summary>
    public static SafeTensors Load(string path) => Parse(File.ReadAllBytes(path), path);

    /// <summary>Async variant: awaits the (multi-hundred-MB) file read instead of blocking on it.</summary>
    public static async Task<SafeTensors> LoadAsync(string path, CancellationToken ct)
        => Parse(await File.ReadAllBytesAsync(path, ct).ConfigureAwait(false), path);

    private static SafeTensors Parse(byte[] bytes, string path)
    {
        if (bytes.Length < 8)
        {
            throw new InvalidDataException($"'{path}' is too small to be a safetensors file.");
        }

        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(0, 8));
        if (headerLen == 0 || (ulong)bytes.Length < 8 + headerLen)
        {
            throw new InvalidDataException($"'{path}' has an invalid safetensors header length ({headerLen}).");
        }

        var headerJson = bytes.AsSpan(8, (int)headerLen);
        var entries = new Dictionary<string, Entry>(StringComparer.Ordinal);

        using (var doc = JsonDocument.Parse(headerJson.ToArray()))
        {
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.NameEquals("__metadata__"))
                {
                    continue;
                }

                var e = prop.Value;
                string dtype = e.GetProperty("dtype").GetString()!;
                var shapeEl = e.GetProperty("shape");
                var shape = new int[shapeEl.GetArrayLength()];
                int idx = 0;
                foreach (var dim in shapeEl.EnumerateArray())
                {
                    shape[idx++] = dim.GetInt32();
                }
                var off = e.GetProperty("data_offsets");
                long begin = off[0].GetInt64();
                long end = off[1].GetInt64();
                entries[prop.Name] = new Entry(dtype, shape, begin, end);
            }
        }

        long dataStart = 8 + (long)headerLen;

        // Guard against a truncated file (e.g. an interrupted download cached under the final
        // name): the header parses fine but the data section is short, which would otherwise fail
        // deep inside Bytes() with a cryptic ArgumentOutOfRangeException. Surface a clear,
        // catchable error so callers can re-download.
        long maxEnd = 0;
        foreach (var e in entries.Values)
        {
            if (e.End > maxEnd)
            {
                maxEnd = e.End;
            }
        }
        if (dataStart + maxEnd > bytes.Length)
        {
            throw new InvalidDataException(
                $"'{path}' is truncated: the header declares {dataStart + maxEnd} bytes but the file is only {bytes.Length}. " +
                "The cached weights file is incomplete - delete it and download again.");
        }

        return new SafeTensors(bytes, dataStart, entries);
    }

    /// <summary>Returns a span over the raw bytes of <paramref name="e"/> within the data section.</summary>
    private ReadOnlySpan<byte> Bytes(in Entry e) => _bytes.AsSpan((int)(_dataStart + e.Begin), (int)(e.End - e.Begin));

    private Entry GetEntry(string name)
    {
        if (!_entries.TryGetValue(name, out var e))
        {
            throw new KeyNotFoundException($"Tensor '{name}' is not present in the safetensors file.");
        }
        return e;
    }

    /// <summary>Returns the raw 16-bit storage of a BF16/F16 tensor without widening - used for the
    /// large embedding table, whose rows are widened lazily one token at a time.</summary>
    public ushort[] ReadRaw16(string name)
    {
        var e = GetEntry(name);
        if (e.Dtype is not ("BF16" or "F16"))
        {
            throw new NotSupportedException($"ReadRaw16 expects a 16-bit tensor; '{name}' is {e.Dtype}.");
        }
        int count = (int)((e.End - e.Begin) / 2);
        var result = new ushort[count];
        var src = Bytes(e);
        for (int i = 0; i < count; i++)
        {
            result[i] = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(i * 2, 2));
        }
        return result;
    }

    /// <summary>Widens a BF16/F16/F32 tensor to a flat <see cref="float"/> array (row-major).</summary>
    public float[] ReadFloat(string name)
    {
        var e = GetEntry(name);
        switch (e.Dtype)
        {
            case "F32":
            {
                int count = (int)((e.End - e.Begin) / 4);
                var result = new float[count];
                var src = Bytes(e);
                for (int i = 0; i < count; i++)
                {
                    result[i] = BinaryPrimitives.ReadSingleLittleEndian(src.Slice(i * 4, 4));
                }
                return result;
            }
            case "BF16":
            {
                int count = (int)((e.End - e.Begin) / 2);
                var result = new float[count];
                var src = Bytes(e);
                for (int i = 0; i < count; i++)
                {
                    result[i] = FloatConversions.BFloat16ToSingle(BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(i * 2, 2)));
                }
                return result;
            }
            case "F16":
            {
                int count = (int)((e.End - e.Begin) / 2);
                var result = new float[count];
                var src = Bytes(e);
                for (int i = 0; i < count; i++)
                {
                    result[i] = FloatConversions.Float16ToSingle(BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(i * 2, 2)));
                }
                return result;
            }
            default:
                throw new NotSupportedException($"Unsupported safetensors dtype '{e.Dtype}' for tensor '{name}'.");
        }
    }
}
