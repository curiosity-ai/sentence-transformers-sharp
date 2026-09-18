#nullable enable

using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace SentenceTransformers.Stq;

/// <summary>
/// Builds an <c>.stq</c> file. Blobs are collected in memory, then laid out back to back (each
/// 64-byte aligned) once every offset is known, and written after the JSON header. A ternary Harrier
/// Small comes to ~60 MB, so holding the payload in memory keeps the writer simple; a converter for a
/// much larger model would want to spool the data section to a temp file instead.
/// </summary>
public sealed class StqWriter
{
    private readonly List<(string Key, byte[] Data)> _blobs = new();
    private readonly Dictionary<string, string> _metadata = new(StringComparer.Ordinal);
    private readonly List<(string Id, HadamardRotation Rotation)> _rotations = new();
    private readonly List<TensorEntry> _tensors = new();

    private sealed record TensorEntry(string Name, StqBand Band, int[] Shape, int GroupSize, string? RotationId, string? CodesKey, string? ScalesKey, string? DataKey);

    public StqWriter()
    {
        _metadata["format"] = "stq";
        _metadata["format_version"] = StqFormat.FormatVersion.ToString();
    }

    /// <summary>Records a free-form header field. Values are stored as strings.</summary>
    public StqWriter SetMetadata(string key, string value)
    {
        _metadata[key] = value;
        return this;
    }

    /// <summary>Registers a rotation that tensors can reference by <paramref name="id"/>.</summary>
    public StqWriter AddRotation(string id, HadamardRotation rotation)
    {
        _rotations.Add((id, rotation));
        _blobs.Add(($"rot:{id}", rotation.SignBits));
        return this;
    }

    /// <summary>Adds a packed tensor (ternary or 4-bit) from already-packed codes and FP16 scales.</summary>
    public StqWriter AddPacked(string name, StqBand band, int[] shape, int groupSize, string? rotationId, byte[] codes, ushort[] scales)
    {
        if (!StqFormat.IsPacked(band))
        {
            throw new ArgumentException($"{StqFormat.BandName(band)} is not a packed band.", nameof(band));
        }

        var scaleBytes = new byte[scales.Length * 2];
        for (int i = 0; i < scales.Length; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(scaleBytes.AsSpan(i * 2, 2), scales[i]);
        }

        _blobs.Add(($"codes:{name}", codes));
        _blobs.Add(($"scales:{name}", scaleBytes));
        _tensors.Add(new TensorEntry(name, band, shape, groupSize, rotationId, $"codes:{name}", $"scales:{name}", null));
        return this;
    }

    /// <summary>Adds a tensor stored verbatim in a float band - for norm vectors and anything else
    /// too small or too sensitive to ternarize.</summary>
    public StqWriter AddRaw(string name, StqBand band, int[] shape, ReadOnlySpan<float> data)
    {
        int bytesPer = StqFormat.RawBytesPerElement(band);
        var buf = new byte[(long)data.Length * bytesPer];
        for (int i = 0; i < data.Length; i++)
        {
            switch (band)
            {
                case StqBand.F32:
                    BinaryPrimitives.WriteSingleLittleEndian(buf.AsSpan(i * 4, 4), data[i]);
                    break;
                case StqBand.F16:
                    BinaryPrimitives.WriteUInt16LittleEndian(buf.AsSpan(i * 2, 2), BitConverter.HalfToUInt16Bits((Half)data[i]));
                    break;
                case StqBand.BF16:
                    // Round-to-nearest-even on the way down to bf16, matching what PyTorch stores.
                    uint bits = BitConverter.SingleToUInt32Bits(data[i]);
                    uint rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
                    BinaryPrimitives.WriteUInt16LittleEndian(buf.AsSpan(i * 2, 2), (ushort)rounded);
                    break;
            }
        }
        _blobs.Add(($"data:{name}", buf));
        _tensors.Add(new TensorEntry(name, band, shape, 0, null, null, null, $"data:{name}"));
        return this;
    }

    /// <summary>Lays the file out and writes it to <paramref name="path"/>.</summary>
    public async Task WriteAsync(string path, CancellationToken ct = default)
    {
        // Pass 1: assign each blob an aligned offset within the data section.
        var offsets = new Dictionary<string, (long Begin, long End)>(StringComparer.Ordinal);
        long cursor = 0;
        foreach (var (key, data) in _blobs)
        {
            cursor = StqFormat.AlignUp(cursor);
            offsets[key] = (cursor, cursor + data.Length);
            cursor += data.Length;
        }
        long dataBytes = cursor;

        // Pass 2: serialize the header now that every offset is known.
        byte[] header = BuildHeader(offsets);
        long headerEnd = 8 + header.Length;
        long dataStart = StqFormat.AlignUp(headerEnd);

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        await using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);

        var prologue = new byte[8];
        Encoding.ASCII.GetBytes(StqFormat.Magic, prologue.AsSpan(0, 4));
        BinaryPrimitives.WriteUInt32LittleEndian(prologue.AsSpan(4, 4), (uint)header.Length);
        await fs.WriteAsync(prologue, ct).ConfigureAwait(false);
        await fs.WriteAsync(header, ct).ConfigureAwait(false);

        var padding = new byte[StqFormat.Alignment];
        await fs.WriteAsync(padding.AsMemory(0, (int)(dataStart - headerEnd)), ct).ConfigureAwait(false);

        long written = 0;
        foreach (var (key, data) in _blobs)
        {
            long begin = offsets[key].Begin;
            if (begin > written)
            {
                await fs.WriteAsync(padding.AsMemory(0, (int)(begin - written)), ct).ConfigureAwait(false);
                written = begin;
            }
            await fs.WriteAsync(data, ct).ConfigureAwait(false);
            written += data.Length;
        }

        if (written != dataBytes)
        {
            throw new InvalidOperationException($"Internal layout error: wrote {written} data bytes, expected {dataBytes}.");
        }
    }

    private byte[] BuildHeader(Dictionary<string, (long Begin, long End)> offsets)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false, SkipValidation = false }))
        {
            w.WriteStartObject();

            w.WriteStartObject("__metadata__");
            foreach (var (k, v) in _metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                w.WriteString(k, v);
            }
            w.WriteEndObject();

            if (_rotations.Count > 0)
            {
                w.WriteStartObject("__rotations__");
                foreach (var (id, rot) in _rotations)
                {
                    w.WriteStartObject(id);
                    w.WriteNumber("dim", rot.Dim);
                    w.WriteNumber("block", rot.Block);
                    WriteRange(w, "signs", offsets[$"rot:{id}"]);
                    w.WriteEndObject();
                }
                w.WriteEndObject();
            }

            foreach (var t in _tensors)
            {
                w.WriteStartObject(t.Name);
                w.WriteString("band", StqFormat.BandName(t.Band));
                w.WriteStartArray("shape");
                foreach (int d in t.Shape) w.WriteNumberValue(d);
                w.WriteEndArray();

                if (StqFormat.IsPacked(t.Band))
                {
                    w.WriteNumber("group_size", t.GroupSize);
                    if (t.RotationId is not null) w.WriteString("rotation", t.RotationId);
                    WriteRange(w, "codes", offsets[t.CodesKey!]);
                    WriteRange(w, "scales", offsets[t.ScalesKey!]);
                }
                else
                {
                    WriteRange(w, "data", offsets[t.DataKey!]);
                }
                w.WriteEndObject();
            }

            w.WriteEndObject();
        }
        return ms.ToArray();
    }

    private static void WriteRange(Utf8JsonWriter w, string name, (long Begin, long End) range)
    {
        w.WriteStartArray(name);
        w.WriteNumberValue(range.Begin);
        w.WriteNumberValue(range.End);
        w.WriteEndArray();
    }
}
