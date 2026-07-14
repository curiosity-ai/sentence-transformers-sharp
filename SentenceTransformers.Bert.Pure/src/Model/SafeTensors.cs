using System.Buffers.Binary;
using System.Text.Json;

namespace SentenceTransformers.Bert.Pure.Model;

/// <summary>
/// Minimal pure-managed reader for the HuggingFace <c>safetensors</c> format: an 8-byte little-endian
/// header length, a JSON header mapping tensor name → {dtype, shape, data_offsets}, then the raw tensor
/// bytes. Only the dtypes the BERT checkpoints use are supported (F32, F16, BF16); integer buffers such
/// as <c>embeddings.position_ids</c> are ignored by callers.
/// </summary>
internal sealed class SafeTensors : IBertWeightProvider
{
    float[] IBertWeightProvider.Read(string name) => ReadFloat(name);

    private readonly byte[] _bytes;
    private readonly long   _dataStart;
    private readonly Dictionary<string, Entry> _entries;

    private readonly record struct Entry(string Dtype, int[] Shape, long Begin, long End);

    private SafeTensors(byte[] bytes, long dataStart, Dictionary<string, Entry> entries)
    {
        _bytes     = bytes;
        _dataStart = dataStart;
        _entries   = entries;
    }

    public bool Contains(string name) => _entries.ContainsKey(name);
    public int[] Shape(string name)   => GetEntry(name).Shape;

    public static SafeTensors Load(string path) => Parse(File.ReadAllBytes(path));
    public static async Task<SafeTensors> LoadAsync(string path, CancellationToken ct)
        => Parse(await File.ReadAllBytesAsync(path, ct).ConfigureAwait(false));

    private static SafeTensors Parse(byte[] bytes)
    {
        if (bytes.Length < 8) throw new InvalidDataException("File too small to be safetensors.");
        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(0, 8));
        if (headerLen == 0 || (ulong)bytes.Length < 8 + headerLen) throw new InvalidDataException("Invalid safetensors header length.");

        var entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        using (var doc = JsonDocument.Parse(bytes.AsMemory(8, (int)headerLen)))
        {
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.NameEquals("__metadata__")) continue;
                var e = prop.Value;
                string dtype = e.GetProperty("dtype").GetString();
                var shapeArr = e.GetProperty("shape");
                var shape = new int[shapeArr.GetArrayLength()];
                for (int i = 0; i < shape.Length; i++) shape[i] = shapeArr[i].GetInt32();
                var off = e.GetProperty("data_offsets");
                entries[prop.Name] = new Entry(dtype, shape, off[0].GetInt64(), off[1].GetInt64());
            }
        }

        long dataStart = 8 + (long)headerLen;

        // Guard against a truncated file (e.g. an interrupted download cached under the final
        // name): the header parses fine but the data section is short, which would otherwise fail
        // deep in ReadFloat with a cryptic ArgumentOutOfRangeException. Surface a clear, catchable
        // error so callers can re-download.
        long maxEnd = 0;
        foreach (var e in entries.Values) if (e.End > maxEnd) maxEnd = e.End;
        if (dataStart + maxEnd > bytes.Length)
            throw new InvalidDataException($"Truncated safetensors: header declares {dataStart + maxEnd} bytes but file is only {bytes.Length}. The cached weights file is incomplete - delete it and download again.");

        return new SafeTensors(bytes, dataStart, entries);
    }

    private Entry GetEntry(string name)
        => _entries.TryGetValue(name, out var e) ? e : throw new KeyNotFoundException($"Tensor '{name}' not found in safetensors.");

    /// <summary>Reads a tensor by name and widens it to <c>float[]</c> (F32/F16/BF16 supported).</summary>
    public float[] ReadFloat(string name)
    {
        var e    = GetEntry(name);
        var span = _bytes.AsSpan((int)(_dataStart + e.Begin), (int)(e.End - e.Begin));

        switch (e.Dtype)
        {
            case "F32":
            {
                int n = span.Length / 4;
                var r = new float[n];
                for (int i = 0; i < n; i++) r[i] = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(i * 4, 4));
                return r;
            }
            case "F16":
            {
                int n = span.Length / 2;
                var r = new float[n];
                for (int i = 0; i < n; i++) r[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(span.Slice(i * 2, 2));
                return r;
            }
            case "BF16":
            {
                int n = span.Length / 2;
                var r = new float[n];
                for (int i = 0; i < n; i++)
                {
                    ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(i * 2, 2));
                    r[i] = BitConverter.Int32BitsToSingle(bits << 16);
                }
                return r;
            }
            default:
                throw new NotSupportedException($"Unsupported safetensors dtype '{e.Dtype}' for tensor '{name}'.");
        }
    }
}
