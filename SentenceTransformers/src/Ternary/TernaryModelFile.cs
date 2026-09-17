#nullable enable

using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace SentenceTransformers.Ternary;

/// <summary>One tensor's entry in an <c>.stq</c> header. Byte ranges are <c>[Begin, End)</c> offsets
/// into the data section.</summary>
public sealed class TernaryTensorInfo
{
    public required string Name { get; init; }
    public required TernaryBand Band { get; init; }
    public required int[] Shape { get; init; }

    /// <summary>Weights per scale group along the last (input) dimension. Zero for raw bands.</summary>
    public int GroupSize { get; init; }

    /// <summary>Id of the rotation whose inverse recovers the original weights, or null when the
    /// tensor is stored in its original basis.</summary>
    public string? RotationId { get; init; }

    /// <summary>Packed code bytes (ternary bands only).</summary>
    public long CodesBegin { get; init; }
    public long CodesEnd { get; init; }

    /// <summary>FP16 group scales, <c>rows * groupsPerRow</c> of them (ternary bands only).</summary>
    public long ScalesBegin { get; init; }
    public long ScalesEnd { get; init; }

    /// <summary>Raw element bytes (float bands only).</summary>
    public long DataBegin { get; init; }
    public long DataEnd { get; init; }

    /// <summary>Total element count across all dimensions.</summary>
    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (int d in Shape) n *= d;
            return n;
        }
    }

    /// <summary>Elements along the last dimension - the axis groups and rotations run along.</summary>
    public int InDim => Shape[^1];

    /// <summary>Rows, i.e. the product of every dimension but the last.</summary>
    public int Rows => (int)(ElementCount / InDim);

    /// <summary>Scale groups per row.</summary>
    public int GroupsPerRow => GroupSize > 0 ? InDim / GroupSize : 0;
}

/// <summary>
/// Reader for the <c>.stq</c> ternary container described in <see cref="TernaryFormat"/>. The whole
/// file is read into memory once (a ternary Harrier Small is ~60 MB) and tensors are decoded lazily,
/// so a runtime can hand the packed bytes straight to a kernel without ever materializing floats.
/// </summary>
public sealed class TernaryModelFile
{
    private readonly byte[] _bytes;
    private readonly long _dataStart;
    private readonly Dictionary<string, TernaryTensorInfo> _tensors;
    private readonly Dictionary<string, TernaryRotation> _rotations;
    private readonly Dictionary<string, string> _metadata;

    private TernaryModelFile(byte[] bytes, long dataStart, Dictionary<string, TernaryTensorInfo> tensors,
                             Dictionary<string, TernaryRotation> rotations, Dictionary<string, string> metadata)
    {
        _bytes = bytes;
        _dataStart = dataStart;
        _tensors = tensors;
        _rotations = rotations;
        _metadata = metadata;
    }

    public IReadOnlyDictionary<string, TernaryTensorInfo> Tensors => _tensors;
    public IReadOnlyDictionary<string, string> Metadata => _metadata;
    public IReadOnlyDictionary<string, TernaryRotation> Rotations => _rotations;

    public bool Contains(string name) => _tensors.ContainsKey(name);

    public static TernaryModelFile Load(string path) => Parse(File.ReadAllBytes(path), path);

    public static async Task<TernaryModelFile> LoadAsync(string path, CancellationToken ct = default)
        => Parse(await File.ReadAllBytesAsync(path, ct).ConfigureAwait(false), path);

    private static TernaryModelFile Parse(byte[] bytes, string path)
    {
        if (bytes.Length < 8 || Encoding.ASCII.GetString(bytes, 0, 4) != TernaryFormat.Magic)
        {
            throw new InvalidDataException($"'{path}' is not an {TernaryFormat.Magic} file.");
        }

        uint headerLen = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4, 4));
        if (headerLen == 0 || 8 + (long)headerLen > bytes.Length)
        {
            throw new InvalidDataException($"'{path}' has an invalid header length ({headerLen}).");
        }

        var tensors   = new Dictionary<string, TernaryTensorInfo>(StringComparer.Ordinal);
        var rotations = new Dictionary<string, TernaryRotation>(StringComparer.Ordinal);
        var metadata  = new Dictionary<string, string>(StringComparer.Ordinal);

        long dataStart = TernaryFormat.AlignUp(8 + headerLen);

        using (var doc = JsonDocument.Parse(bytes.AsMemory(8, (int)headerLen)))
        {
            var root = doc.RootElement;

            if (root.TryGetProperty("__metadata__", out var meta))
            {
                foreach (var p in meta.EnumerateObject())
                {
                    metadata[p.Name] = p.Value.ToString();
                }
            }

            if (metadata.TryGetValue("format_version", out var fv) && int.TryParse(fv, out int v) && v > TernaryFormat.FormatVersion)
            {
                throw new InvalidDataException(
                    $"'{path}' was written by a newer format (version {v}); this build reads up to {TernaryFormat.FormatVersion}.");
            }

            // Rotations are parsed first: a tensor entry may reference one by id.
            if (root.TryGetProperty("__rotations__", out var rots))
            {
                foreach (var p in rots.EnumerateObject())
                {
                    var r = p.Value;
                    int dim = r.GetProperty("dim").GetInt32();
                    int block = r.GetProperty("block").GetInt32();
                    var span = r.GetProperty("signs");
                    long b = span[0].GetInt64(), e = span[1].GetInt64();
                    var signs = bytes.AsSpan((int)(dataStart + b), (int)(e - b)).ToArray();
                    rotations[p.Name] = new TernaryRotation(dim, block, signs);
                }
            }

            foreach (var p in root.EnumerateObject())
            {
                if (p.Name.StartsWith("__", StringComparison.Ordinal))
                {
                    continue;
                }

                var e = p.Value;
                var band = TernaryFormat.ParseBand(e.GetProperty("band").GetString()!);
                var shapeEl = e.GetProperty("shape");
                var shape = new int[shapeEl.GetArrayLength()];
                int i = 0;
                foreach (var d in shapeEl.EnumerateArray())
                {
                    shape[i++] = d.GetInt32();
                }

                string? rotationId = e.TryGetProperty("rotation", out var rid) && rid.ValueKind == JsonValueKind.String ? rid.GetString() : null;
                if (rotationId is not null && !rotations.ContainsKey(rotationId))
                {
                    throw new InvalidDataException($"Tensor '{p.Name}' references unknown rotation '{rotationId}'.");
                }

                var info = TernaryFormat.IsTernary(band)
                    ? new TernaryTensorInfo
                    {
                        Name = p.Name,
                        Band = band,
                        Shape = shape,
                        GroupSize = e.GetProperty("group_size").GetInt32(),
                        RotationId = rotationId,
                        CodesBegin = e.GetProperty("codes")[0].GetInt64(),
                        CodesEnd = e.GetProperty("codes")[1].GetInt64(),
                        ScalesBegin = e.GetProperty("scales")[0].GetInt64(),
                        ScalesEnd = e.GetProperty("scales")[1].GetInt64(),
                    }
                    : new TernaryTensorInfo
                    {
                        Name = p.Name,
                        Band = band,
                        Shape = shape,
                        RotationId = rotationId,
                        DataBegin = e.GetProperty("data")[0].GetInt64(),
                        DataEnd = e.GetProperty("data")[1].GetInt64(),
                    };

                tensors[p.Name] = info;
            }
        }

        // A truncated download parses fine but reads past the end later, deep inside a kernel. Fail
        // here instead, with something the caller can act on.
        long maxEnd = 0;
        foreach (var t in tensors.Values)
        {
            maxEnd = Math.Max(maxEnd, Math.Max(t.CodesEnd, Math.Max(t.ScalesEnd, t.DataEnd)));
        }
        if (dataStart + maxEnd > bytes.Length)
        {
            throw new InvalidDataException(
                $"'{path}' is truncated: the header declares {dataStart + maxEnd} bytes but the file is only {bytes.Length}. " +
                "Delete the cached file and download it again.");
        }

        return new TernaryModelFile(bytes, dataStart, tensors, rotations, metadata);
    }

    public TernaryTensorInfo Info(string name)
        => _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"Tensor '{name}' is not present in the ternary model file.");

    public int[] Shape(string name) => Info(name).Shape;

    /// <summary>The rotation a tensor was stored under, or null if it is in its original basis.</summary>
    public TernaryRotation? RotationFor(TernaryTensorInfo info)
        => info.RotationId is null ? null : _rotations[info.RotationId];

    /// <summary>The packed code bytes, referenced in place (no copy).</summary>
    public ReadOnlyMemory<byte> Codes(TernaryTensorInfo info)
    {
        RequireTernary(info);
        return _bytes.AsMemory((int)(_dataStart + info.CodesBegin), (int)(info.CodesEnd - info.CodesBegin));
    }

    /// <summary>The FP16 group scales, widened to float (one per group, <c>Rows * GroupsPerRow</c>).</summary>
    public float[] Scales(TernaryTensorInfo info)
    {
        RequireTernary(info);
        int count = (int)((info.ScalesEnd - info.ScalesBegin) / 2);
        var src = _bytes.AsSpan((int)(_dataStart + info.ScalesBegin), count * 2);
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(i * 2, 2)));
        }
        return result;
    }

    /// <summary>
    /// Fully decodes a tensor to float32 in its <b>original</b> basis: unpacks the codes, applies the
    /// group scales and, when the tensor was stored rotated, un-rotates every row. This is the
    /// reference path - validation and any consumer that just wants the weights back. Runtime kernels
    /// use <see cref="Codes"/>/<see cref="Scales"/> and leave the rotation on the activation instead.
    /// </summary>
    public float[] ReadFloat(string name)
    {
        var info = Info(name);
        long n = info.ElementCount;
        var result = new float[n];

        if (!TernaryFormat.IsTernary(info.Band))
        {
            var src = _bytes.AsSpan((int)(_dataStart + info.DataBegin), (int)(info.DataEnd - info.DataBegin));
            switch (info.Band)
            {
                case TernaryBand.F32:
                    for (long i = 0; i < n; i++) result[i] = BinaryPrimitives.ReadSingleLittleEndian(src.Slice((int)i * 4, 4));
                    break;
                case TernaryBand.F16:
                    for (long i = 0; i < n; i++) result[i] = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(src.Slice((int)i * 2, 2)));
                    break;
                case TernaryBand.BF16:
                    for (long i = 0; i < n; i++) result[i] = BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(src.Slice((int)i * 2, 2)) << 16);
                    break;
            }
            return result;
        }

        var codes = Codes(info).Span;
        var scales = Scales(info);
        var rotation = RotationFor(info);
        int inDim = info.InDim, groups = info.GroupsPerRow, gs = info.GroupSize;
        int groupBytes = TernaryFormat.CodeBytesPerGroup(info.Band, gs);
        int rowBytes = groups * groupBytes;

        for (int r = 0; r < info.Rows; r++)
        {
            var row = result.AsSpan(r * inDim, inDim);
            for (int g = 0; g < groups; g++)
            {
                TernaryPacking.UnpackGroupScaled(info.Band, codes.Slice(r * rowBytes + g * groupBytes, groupBytes),
                                                 row.Slice(g * gs, gs), gs, scales[r * groups + g]);
            }
            rotation?.ApplyInverse(row);
        }
        return result;
    }

    private static void RequireTernary(TernaryTensorInfo info)
    {
        if (!TernaryFormat.IsTernary(info.Band))
        {
            throw new InvalidOperationException($"Tensor '{info.Name}' is stored as {TernaryFormat.BandName(info.Band)}, not a ternary band.");
        }
    }

    /// <summary>Total bytes across every tensor blob - what the file costs in memory once loaded.</summary>
    public long PayloadBytes
    {
        get
        {
            long total = 0;
            foreach (var t in _tensors.Values)
            {
                total += (t.CodesEnd - t.CodesBegin) + (t.ScalesEnd - t.ScalesBegin) + (t.DataEnd - t.DataBegin);
            }
            return total;
        }
    }
}
