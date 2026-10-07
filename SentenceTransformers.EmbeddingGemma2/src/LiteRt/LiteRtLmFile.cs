using System.Buffers.Binary;
using System.Text;

namespace SentenceTransformers.EmbeddingGemma2.LiteRt;

/// <summary>Section payload kinds of a LiteRT-LM bundle (<c>AnySectionDataType</c> in the header schema).</summary>
internal enum LiteRtLmSectionType : byte
{
    None = 0,
    GenericBinaryData = 1,
    Deprecated = 2,
    TFLiteModel = 3,
    SP_Tokenizer = 4,
    LlmMetadataProto = 5,
    HF_Tokenizer_Zlib = 6,
    TFLiteWeights = 7,
    EmbeddingMetadataProto = 8,
    ExecutorMetadataProto = 9,
    TtsMetadataProto = 10,
    AsrMetadataProto = 11,
}

/// <summary>One section of a <c>.litertlm</c> file: a byte range plus its type and the <c>model_type</c>
/// key (e.g. <c>tf_lite_text_encoder</c>) that pairs a TFLite graph with its external weight section.</summary>
internal sealed record LiteRtLmSection(int Index, LiteRtLmSectionType Type, string ModelType, long Begin, long End)
{
    public long Length => End - Begin;
}

/// <summary>
/// Reader for the LiteRT-LM container format (<c>.litertlm</c>): an 8-byte <c>LITERTLM</c> magic, a
/// version triple, a FlatBuffers header listing the sections, then 16 KB-aligned section payloads
/// (tokenizer, metadata protos, TFLite graphs and their external weight blobs).
/// Sections are read lazily with positional reads, so opening the 485 MB multimodal bundle only
/// touches the parts a given modality needs.
/// </summary>
internal sealed class LiteRtLmFile : IDisposable
{
    private static ReadOnlySpan<byte> Magic => "LITERTLM"u8;
    private const int HeaderEndLocationOffset = 24;
    private const int HeaderBeginOffset = 32;

    private readonly Microsoft.Win32.SafeHandles.SafeFileHandle _handle;

    public string Path { get; }
    public (int Major, int Minor, int Patch) Version { get; }
    public IReadOnlyList<LiteRtLmSection> Sections { get; }
    public IReadOnlyDictionary<string, string> SystemMetadata { get; }

    private LiteRtLmFile(string path, Microsoft.Win32.SafeHandles.SafeFileHandle handle, (int, int, int) version, List<LiteRtLmSection> sections, Dictionary<string, string> systemMetadata)
    {
        Path = path;
        _handle = handle;
        Version = version;
        Sections = sections;
        SystemMetadata = systemMetadata;
    }

    public static LiteRtLmFile Open(string path)
    {
        var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
        try
        {
            var head = new byte[HeaderBeginOffset];
            ReadExactly(handle, head, 0);
            if (!head.AsSpan(0, 8).SequenceEqual(Magic))
            {
                throw new InvalidDataException($"'{path}' is not a LiteRT-LM file (bad magic).");
            }
            var version = ((int)BinaryPrimitives.ReadUInt32LittleEndian(head.AsSpan(8)),
                           (int)BinaryPrimitives.ReadUInt32LittleEndian(head.AsSpan(12)),
                           (int)BinaryPrimitives.ReadUInt32LittleEndian(head.AsSpan(16)));
            long headerEnd = (long)BinaryPrimitives.ReadUInt64LittleEndian(head.AsSpan(HeaderEndLocationOffset));
            long fileLength = RandomAccess.GetLength(handle);
            if (headerEnd <= HeaderBeginOffset || headerEnd > fileLength)
            {
                throw new InvalidDataException($"'{path}' has a corrupt LiteRT-LM header.");
            }

            var header = new byte[headerEnd - HeaderBeginOffset];
            ReadExactly(handle, header, HeaderBeginOffset);
            var root = FbTable.Root(header);

            var system = new Dictionary<string, string>(StringComparer.Ordinal);
            var sys = root.GetTable(4);
            if (!sys.IsNull)
            {
                int n = sys.VectorLength(4);
                for (int i = 0; i < n; i++)
                {
                    var kv = sys.GetTableElement(4, i);
                    system[kv.GetString(4) ?? ""] = ReadKeyValueString(kv);
                }
            }

            var sections = new List<LiteRtLmSection>();
            var secMeta = root.GetTable(6);
            int count = secMeta.IsNull ? 0 : secMeta.VectorLength(4);
            for (int i = 0; i < count; i++)
            {
                var obj = secMeta.GetTableElement(4, i);
                string modelType = null;
                int items = obj.VectorLength(4);
                for (int j = 0; j < items; j++)
                {
                    var kv = obj.GetTableElement(4, j);
                    if (kv.GetString(4) == "model_type")
                    {
                        modelType = ReadKeyValueString(kv);
                    }
                }
                long begin = (long)obj.GetUInt64(6);
                long end = (long)obj.GetUInt64(8);
                if (begin < 0 || end < begin || end > fileLength)
                {
                    throw new InvalidDataException($"'{path}' section {i} is out of range (file truncated?).");
                }
                sections.Add(new LiteRtLmSection(i, (LiteRtLmSectionType)obj.GetByte(10), modelType, begin, end));
            }
            return new LiteRtLmFile(path, handle, version, sections, system);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>Reads a <c>StringValue</c> union value of a header <c>KeyValuePair</c> (other value types yield null).</summary>
    private static string ReadKeyValueString(FbTable kv)
    {
        const byte StringValueType = 9;
        if (kv.GetByte(6) != StringValueType)
        {
            return null;
        }
        var value = kv.GetTable(8);
        return value.IsNull ? null : value.GetString(4);
    }

    private static void ReadExactly(Microsoft.Win32.SafeHandles.SafeFileHandle handle, Span<byte> buffer, long offset)
    {
        while (buffer.Length > 0)
        {
            int n = RandomAccess.Read(handle, buffer, offset);
            if (n <= 0)
            {
                throw new EndOfStreamException("Unexpected end of LiteRT-LM file.");
            }
            buffer = buffer.Slice(n);
            offset += n;
        }
    }

    public LiteRtLmSection Find(LiteRtLmSectionType type, string modelType = null)
        => Sections.FirstOrDefault(s => s.Type == type && (modelType is null || s.ModelType == modelType));

    public bool HasModel(string modelType) => Find(LiteRtLmSectionType.TFLiteModel, modelType) is not null;

    public byte[] ReadSection(LiteRtLmSection section)
    {
        ArgumentNullException.ThrowIfNull(section);
        var data = GC.AllocateUninitializedArray<byte>(checked((int)section.Length));
        ReadExactly(_handle, data, section.Begin);
        return data;
    }

    public byte[] ReadSection(LiteRtLmSectionType type, string modelType = null)
    {
        var s = Find(type, modelType) ?? throw new InvalidDataException($"Section {type} {modelType} not found in '{Path}'.");
        return ReadSection(s);
    }

    /// <summary>Loads a TFLite graph together with its external weight section (same <c>model_type</c>).</summary>
    public TfLiteModel ReadModel(string modelType)
    {
        var graph = ReadSection(LiteRtLmSectionType.TFLiteModel, modelType);
        var weightsSection = Find(LiteRtLmSectionType.TFLiteWeights, modelType);
        var weights = weightsSection is null ? null : ReadSection(weightsSection);
        return new TfLiteModel(graph, weights);
    }

    public EmbeddingMetadata ReadEmbeddingMetadata()
        => EmbeddingMetadata.Parse(ReadSection(LiteRtLmSectionType.EmbeddingMetadataProto));

    public void Dispose() => _handle.Dispose();

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"LiteRT-LM {Version.Major}.{Version.Minor}.{Version.Patch}: {Path}");
        foreach (var s in Sections)
        {
            sb.AppendLine($"  [{s.Index}] {s.Type} {s.ModelType} {s.Length:N0} bytes");
        }
        return sb.ToString();
    }
}

/// <summary>The subset of the LiteRT-LM <c>EmbeddingMetadata</c> proto this port uses.</summary>
internal sealed class EmbeddingMetadata
{
    public int BosId { get; private set; } = 2;
    public int EosId { get; private set; } = 1;
    public int MinInputLength { get; private set; }
    public int MaxInputLength { get; private set; }
    public string StartOfImageToken { get; private set; }
    public string EndOfImageToken { get; private set; }
    public string StartOfAudioToken { get; private set; }
    public string EndOfAudioToken { get; private set; }
    public int PatchWidth { get; private set; }
    public int PatchHeight { get; private set; }
    public int MaxNumPatches { get; private set; }
    public int PoolingKernelSize { get; private set; }
    public AudioPreprocessorConfig Audio { get; private set; }

    public static EmbeddingMetadata Parse(ReadOnlySpan<byte> data)
    {
        var m = new EmbeddingMetadata();
        var r = new ProtoReader(data);
        while (r.Next(out int f, out int wt))
        {
            switch (f)
            {
                case 1: m.ParseModelType(r.ReadBytes()); break;
                case 2: m.BosId = ParseTokenId(r.ReadBytes(), m.BosId); break;
                case 3: m.EosId = ParseTokenId(r.ReadBytes(), m.EosId); break;
                case 4: m.Audio = ParseAudio(r.ReadBytes()); break;
                case 5: m.MinInputLength = r.ReadInt32(); break;
                case 6: m.MaxInputLength = r.ReadInt32(); break;
                default: r.Skip(wt); break;
            }
        }
        return m;
    }

    private void ParseModelType(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        while (r.Next(out int f, out int wt))
        {
            if (f != 2)
            {
                r.Skip(wt);
                continue;
            }
            var g = new ProtoReader(r.ReadBytes());
            while (g.Next(out int gf, out int gwt))
            {
                switch (gf)
                {
                    case 1: StartOfImageToken = ParseTokenString(g.ReadBytes()); break;
                    case 2: EndOfImageToken = ParseTokenString(g.ReadBytes()); break;
                    case 3: PatchWidth = g.ReadInt32(); break;
                    case 4: PatchHeight = g.ReadInt32(); break;
                    case 5: MaxNumPatches = g.ReadInt32(); break;
                    case 6: PoolingKernelSize = g.ReadInt32(); break;
                    case 7: StartOfAudioToken = ParseTokenString(g.ReadBytes()); break;
                    case 8: EndOfAudioToken = ParseTokenString(g.ReadBytes()); break;
                    default: g.Skip(gwt); break;
                }
            }
        }
    }

    private static int ParseTokenId(ReadOnlySpan<byte> data, int fallback)
    {
        var r = new ProtoReader(data);
        while (r.Next(out int f, out int wt))
        {
            if (f == 1)
            {
                var ids = new ProtoReader(r.ReadBytes());
                while (ids.Next(out int idf, out int idwt))
                {
                    if (idf == 1)
                    {
                        if (idwt == 2)
                        {
                            var packed = new ProtoReader(ids.ReadBytes());
                            return packed.ReadInt32();
                        }
                        return ids.ReadInt32();
                    }
                    ids.Skip(idwt);
                }
            }
            else
            {
                r.Skip(wt);
            }
        }
        return fallback;
    }

    private static string ParseTokenString(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        while (r.Next(out int f, out int wt))
        {
            if (f == 2)
            {
                return r.ReadString();
            }
            r.Skip(wt);
        }
        return null;
    }

    private static AudioPreprocessorConfig ParseAudio(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        while (r.Next(out int f, out int wt))
        {
            if (f != 1)
            {
                r.Skip(wt);
                continue;
            }
            var c = new AudioPreprocessorConfig();
            var a = new ProtoReader(r.ReadBytes());
            while (a.Next(out int af, out int awt))
            {
                switch (af)
                {
                    case 1: c.SampleRateHz = a.ReadInt32(); break;
                    case 2: c.NumChannels = a.ReadInt32(); break;
                    case 3: c.FrameLength = a.ReadInt32(); break;
                    case 4: c.HopLength = a.ReadInt32(); break;
                    case 5: c.FftLength = a.ReadInt32(); break;
                    case 6: c.InputScale = a.ReadFloat(); break;
                    case 7: c.PreEmphasisFactor = a.ReadFloat(); break;
                    case 8: c.NumMelBins = a.ReadInt32(); break;
                    case 9: c.MelLowHz = a.ReadFloat(); break;
                    case 10: c.MelHighHz = a.ReadFloat(); break;
                    case 11: c.MelFloor = a.ReadFloat(); break;
                    case 12: c.NormalizeMel = a.ReadBool(); break;
                    case 13: c.AddFloorToMelBeforeLog = a.ReadBool(); break;
                    case 14: c.SemicausalPadding = a.ReadBool(); break;
                    case 15: c.NonZeroHanning = a.ReadBool(); break;
                    case 16: c.PeriodicHanning = a.ReadBool(); break;
                    case 17: c.FftPaddingType = a.ReadInt32(); break;
                    default: a.Skip(awt); break;
                }
            }
            return c;
        }
        return null;
    }
}

/// <summary>Audio front-end parameters (<c>MiniAudioPreprocessorConfig</c>).</summary>
internal sealed class AudioPreprocessorConfig
{
    public int SampleRateHz { get; set; }
    public int NumChannels { get; set; }
    public int FrameLength { get; set; }
    public int HopLength { get; set; }
    public int FftLength { get; set; }
    public float InputScale { get; set; } = 1f;
    public float PreEmphasisFactor { get; set; }
    public int NumMelBins { get; set; }
    public float MelLowHz { get; set; }
    public float MelHighHz { get; set; }
    public float MelFloor { get; set; }
    public bool NormalizeMel { get; set; }
    public bool AddFloorToMelBeforeLog { get; set; }
    public bool SemicausalPadding { get; set; }
    public bool NonZeroHanning { get; set; }
    public bool PeriodicHanning { get; set; }
    public int FftPaddingType { get; set; }
}
