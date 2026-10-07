using System.Text;
using SentenceTransformers.EmbeddingGemma2.LiteRt;

namespace SentenceTransformers.EmbeddingGemma2.Tokenizer;

/// <summary>One emitted token: vocabulary id plus the UTF-16 span <c>[Start, Start+Length)</c> of the
/// source text it covers. Byte-fallback tokens of a multi-byte character share the character's span:
/// the first byte token carries it and the remaining ones have <c>Length == 0</c>.</summary>
internal readonly record struct SpToken(int Id, int Start, int Length)
{
    public int End => Start + Length;
}

/// <summary>
/// Pure-managed port of SentencePiece's BPE encoder (<c>bpe_model.cc</c>) driven directly by the
/// <c>ModelProto</c> stored in the LiteRT-LM bundle. Matches the reference library for the Gemma 4
/// tokenizer used by EmbeddingGemma 2: identity normalizer (spaces escaped to <c>U+2581</c>, no dummy
/// prefix), user-defined symbols matched atomically (longest match), merges applied by descending
/// piece score with leftmost tie-break, and UTF-8 byte fallback (<c>&lt;0xNN&gt;</c>) for symbols
/// outside the vocabulary.
/// </summary>
internal sealed class SentencePieceBpe
{
    private enum PieceType : byte { Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6 }

    private const char Metaspace = '▁';

    private readonly string[] _pieces;
    private readonly float[] _scores;
    private readonly PieceType[] _types;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _normal;   // NORMAL/USER_DEFINED/UNUSED
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _reserved; // CONTROL/UNKNOWN/BYTE
    private readonly int[] _byteIds = new int[256];
    private readonly TrieNode _userDefined = new();
    private readonly bool _byteFallback;
    private readonly bool _addDummyPrefix;
    private readonly bool _escapeWhitespaces;
    private readonly bool _removeExtraWhitespaces;

    public int UnkId { get; }
    public int BosId { get; }
    public int EosId { get; }
    public int PadId { get; }
    public int VocabSize => _pieces.Length;

    private sealed class TrieNode
    {
        public Dictionary<char, TrieNode> Next;
        public bool Terminal;
    }

    private SentencePieceBpe(List<(string Piece, float Score, PieceType Type)> pieces, int modelType, bool byteFallback, int unkId, int bosId, int eosId, int padId,
                             bool addDummyPrefix, bool escapeWhitespaces, bool removeExtraWhitespaces, string normalizerName, int charsMapLength)
    {
        const int Bpe = 2;
        if (modelType != Bpe)
        {
            throw new NotSupportedException($"Only SentencePiece BPE models are supported (model_type={modelType}).");
        }
        if (charsMapLength > 0 || (normalizerName is not null && normalizerName != "identity"))
        {
            throw new NotSupportedException($"SentencePiece normalizer '{normalizerName}' with a precompiled charsmap is not supported (only 'identity').");
        }

        int n = pieces.Count;
        _pieces = new string[n];
        _scores = new float[n];
        _types = new PieceType[n];
        var normal = new Dictionary<string, int>(n, StringComparer.Ordinal);
        var reserved = new Dictionary<string, int>(1024, StringComparer.Ordinal);
        Array.Fill(_byteIds, -1);
        for (int i = 0; i < n; i++)
        {
            var (piece, score, type) = pieces[i];
            _pieces[i] = piece;
            _scores[i] = score;
            _types[i] = type;
            if (type is PieceType.Normal or PieceType.UserDefined or PieceType.Unused)
            {
                normal.TryAdd(piece, i);
            }
            else
            {
                reserved.TryAdd(piece, i);
            }
            if (type == PieceType.UserDefined)
            {
                AddUserDefined(piece);
            }
            if (type == PieceType.Byte && piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece[5] == '>')
            {
                _byteIds[Convert.ToInt32(piece.Substring(3, 2), 16)] = i;
            }
        }
        _normal = normal.GetAlternateLookup<ReadOnlySpan<char>>();
        _reserved = reserved.GetAlternateLookup<ReadOnlySpan<char>>();
        _byteFallback = byteFallback;
        _addDummyPrefix = addDummyPrefix;
        _escapeWhitespaces = escapeWhitespaces;
        _removeExtraWhitespaces = removeExtraWhitespaces;
        UnkId = unkId;
        BosId = bosId;
        EosId = eosId;
        PadId = padId;
        if (_removeExtraWhitespaces)
        {
            // Not used by Gemma tokenizers; collapsing whitespace would break the 1:1 offset mapping below.
            throw new NotSupportedException("SentencePiece remove_extra_whitespaces=true is not supported.");
        }
    }

    private void AddUserDefined(string piece)
    {
        var node = _userDefined;
        foreach (var c in piece)
        {
            node.Next ??= new Dictionary<char, TrieNode>();
            if (!node.Next.TryGetValue(c, out var child))
            {
                child = new TrieNode();
                node.Next[c] = child;
            }
            node = child;
        }
        node.Terminal = true;
    }

    /// <summary>Parses a serialized SentencePiece <c>ModelProto</c>.</summary>
    public static SentencePieceBpe FromModelProto(ReadOnlySpan<byte> proto)
    {
        var pieces = new List<(string, float, PieceType)>(270_000);
        // proto2 defaults.
        int modelType = 1, unkId = 0, bosId = 1, eosId = 2, padId = -1;
        bool byteFallback = false, addDummyPrefix = true, escapeWs = true, removeExtraWs = true;
        string normalizerName = null;
        int charsMapLength = 0;

        var r = new ProtoReader(proto);
        while (r.Next(out int field, out int wt))
        {
            switch (field)
            {
                case 1:
                {
                    var p = new ProtoReader(r.ReadBytes());
                    string piece = "";
                    float score = 0;
                    var type = PieceType.Normal;
                    while (p.Next(out int pf, out int pwt))
                    {
                        switch (pf)
                        {
                            case 1: piece = p.ReadString(); break;
                            case 2: score = p.ReadFloat(); break;
                            case 3: type = (PieceType)p.ReadInt32(); break;
                            default: p.Skip(pwt); break;
                        }
                    }
                    pieces.Add((piece, score, type));
                    break;
                }
                case 2:
                {
                    var t = new ProtoReader(r.ReadBytes());
                    while (t.Next(out int tf, out int twt))
                    {
                        switch (tf)
                        {
                            case 3: modelType = t.ReadInt32(); break;
                            case 35: byteFallback = t.ReadBool(); break;
                            case 40: unkId = t.ReadInt32(); break;
                            case 41: bosId = t.ReadInt32(); break;
                            case 42: eosId = t.ReadInt32(); break;
                            case 43: padId = t.ReadInt32(); break;
                            default: t.Skip(twt); break;
                        }
                    }
                    break;
                }
                case 3:
                {
                    var nrm = new ProtoReader(r.ReadBytes());
                    while (nrm.Next(out int nf, out int nwt))
                    {
                        switch (nf)
                        {
                            case 1: normalizerName = nrm.ReadString(); break;
                            case 2: charsMapLength = nrm.ReadBytes().Length; break;
                            case 3: addDummyPrefix = nrm.ReadBool(); break;
                            case 4: removeExtraWs = nrm.ReadBool(); break;
                            case 5: escapeWs = nrm.ReadBool(); break;
                            default: nrm.Skip(nwt); break;
                        }
                    }
                    break;
                }
                default:
                    r.Skip(wt);
                    break;
            }
        }
        return new SentencePieceBpe(pieces, modelType, byteFallback, unkId, bosId, eosId, padId, addDummyPrefix, escapeWs, removeExtraWs, normalizerName, charsMapLength);
    }

    public string IdToPiece(int id) => (uint)id < (uint)_pieces.Length ? _pieces[id] : null;

    /// <summary>Returns the id of a piece (control/byte pieces first, then normal ones), or <see cref="UnkId"/>.</summary>
    public int PieceToId(ReadOnlySpan<char> piece)
    {
        if (_reserved.TryGetValue(piece, out int id))
        {
            return id;
        }
        return _normal.TryGetValue(piece, out id) ? id : UnkId;
    }

    /// <summary>Decodes ids back to text (byte pieces reassembled as UTF-8, metaspace -> space, control tokens dropped).</summary>
    public string Decode(ReadOnlySpan<int> ids)
    {
        var bytes = new List<byte>();
        var sb = new StringBuilder();
        void Flush()
        {
            if (bytes.Count > 0)
            {
                sb.Append(Encoding.UTF8.GetString(bytes.ToArray()));
                bytes.Clear();
            }
        }
        foreach (var id in ids)
        {
            if ((uint)id >= (uint)_pieces.Length)
            {
                continue;
            }
            switch (_types[id])
            {
                case PieceType.Control:
                    continue;
                case PieceType.Byte:
                    bytes.Add(Convert.ToByte(_pieces[id].Substring(3, 2), 16));
                    continue;
                default:
                    Flush();
                    sb.Append(_pieces[id].Replace(Metaspace, ' '));
                    break;
            }
        }
        Flush();
        return sb.ToString();
    }

    private struct Symbol
    {
        public int Prev;
        public int Next;
        public int Start;   // UTF-16 index into the normalized text
        public int Length;  // UTF-16 length; 0 once merged away
        public bool Freeze;
    }

    private readonly record struct Candidate(int Left, int Right, int Length);

    /// <summary>Encodes text to SentencePiece tokens (no BOS/EOS) with source offsets.</summary>
    public List<SpToken> Encode(string text)
    {
        var result = new List<SpToken>();
        if (string.IsNullOrEmpty(text))
        {
            return result;
        }

        // Normalize: identity + whitespace escaping. Both substitutions are 1:1 in UTF-16 units, so
        // normalized indices map straight back onto the source text (offset == index - prefixShift).
        int prefixShift = _addDummyPrefix ? 1 : 0;
        var normalized = new char[text.Length + prefixShift];
        if (_addDummyPrefix)
        {
            normalized[0] = _escapeWhitespaces ? Metaspace : ' ';
        }
        for (int k = 0; k < text.Length; k++)
        {
            char c = text[k];
            normalized[k + prefixShift] = _escapeWhitespaces && c == ' ' ? Metaspace : c;
        }
        var norm = normalized.AsSpan();

        // Split into initial symbols: user-defined symbols (longest match, frozen) or single code points.
        var symbols = new List<Symbol>(norm.Length);
        int pos = 0;
        while (pos < norm.Length)
        {
            int udLen = MatchUserDefined(norm, pos);
            bool freeze = udLen > 0;
            int len = freeze ? udLen : (char.IsHighSurrogate(norm[pos]) && pos + 1 < norm.Length && char.IsLowSurrogate(norm[pos + 1]) ? 2 : 1);
            symbols.Add(new Symbol { Prev = symbols.Count - 1, Next = pos + len < norm.Length ? symbols.Count + 1 : -1, Start = pos, Length = len, Freeze = freeze });
            pos += len;
        }

        var syms = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(symbols);
        // Max-score first; ties broken by the leftmost position (SentencePiece's agenda ordering).
        var agenda = new PriorityQueue<Candidate, (float NegScore, int Left)>();

        for (int k = 1; k < syms.Length; k++)
        {
            TryAdd(syms, norm, agenda, k - 1, k);
        }

        while (agenda.TryDequeue(out var top, out _))
        {
            ref var l = ref syms[top.Left];
            ref var r = ref syms[top.Right];
            // Skip stale candidates whose symbols changed since they were queued.
            if (l.Length == 0 || r.Length == 0 || l.Length + r.Length != top.Length)
            {
                continue;
            }
            l.Length += r.Length;
            r.Length = 0;
            l.Next = r.Next;
            if (r.Next >= 0)
            {
                syms[r.Next].Prev = top.Left;
            }
            TryAdd(syms, norm, agenda, l.Prev, top.Left);
            TryAdd(syms, norm, agenda, top.Left, l.Next);
        }

        // Walk the surviving symbols (symbol 0 is never merged away: merges always keep the left one).
        Span<byte> utf8 = stackalloc byte[8];
        int i = 0;
        while (i >= 0 && i < syms.Length)
        {
            ref var s = ref syms[i];
            var piece = norm.Slice(s.Start, s.Length);
            int srcStart = Math.Max(0, s.Start - prefixShift);
            int srcLen = Math.Max(0, s.Start + s.Length - prefixShift - srcStart);
            int id = PieceToId(piece);
            if (id == UnkId && _byteFallback)
            {
                // Byte fallback: one <0xNN> token per UTF-8 byte of every code point in the piece.
                int cp = 0;
                while (cp < piece.Length)
                {
                    int cpLen = char.IsHighSurrogate(piece[cp]) && cp + 1 < piece.Length ? 2 : 1;
                    ReadOnlySpan<char> text16 = piece.Slice(cp, cpLen);
                    if (text16.Length == 1 && text16[0] == Metaspace && _escapeWhitespaces)
                    {
                        // Metaspace stands for a space in the source; it is always in the vocab, but be safe.
                        text16 = " ";
                    }
                    int nb = Encoding.UTF8.GetBytes(text16, utf8);
                    int charStart = Math.Max(0, s.Start + cp - prefixShift);
                    for (int b = 0; b < nb; b++)
                    {
                        int bid = _byteIds[utf8[b]];
                        result.Add(new SpToken(bid >= 0 ? bid : UnkId, charStart, b == 0 ? cpLen : 0));
                    }
                    cp += cpLen;
                }
            }
            else
            {
                result.Add(new SpToken(id, srcStart, srcLen));
            }
            i = s.Next;
        }
        return result;
    }

    private void TryAdd(Span<Symbol> s, ReadOnlySpan<char> norm, PriorityQueue<Candidate, (float NegScore, int Left)> agenda, int left, int right)
    {
        if (left < 0 || right < 0 || s[left].Freeze || s[right].Freeze)
        {
            return;
        }
        var piece = norm.Slice(s[left].Start, s[left].Length + s[right].Length);
        if (_normal.TryGetValue(piece, out int id))
        {
            agenda.Enqueue(new Candidate(left, right, piece.Length), (-_scores[id], left));
        }
    }

    private int MatchUserDefined(ReadOnlySpan<char> text, int pos)
    {
        var node = _userDefined;
        int best = 0;
        for (int i = pos; i < text.Length; i++)
        {
            if (node.Next is null || !node.Next.TryGetValue(text[i], out node))
            {
                break;
            }
            if (node.Terminal)
            {
                best = i - pos + 1;
            }
        }
        return best;
    }
}
