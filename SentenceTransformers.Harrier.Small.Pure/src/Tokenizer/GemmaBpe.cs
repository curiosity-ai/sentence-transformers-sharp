using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using UID;

namespace SentenceTransformers.Harrier.Small.Pure.Tokenizer;

/// <summary>One emitted token: its vocabulary id and the <c>[Start, Start+Length)</c> character span
/// (UTF-16 indices) it covers in <see cref="Source"/>. The token never owns a substring of the input -
/// callers slice <see cref="Source"/> on demand via <see cref="AsSpan"/> (and look the surface form up
/// via <see cref="GemmaBpe.IdToToken"/> when they need it).</summary>
internal readonly record struct BpeToken(int Id, string Source, int Start, int Length)
{
    public int End => Start + Length;

    /// <summary>The slice of <see cref="Source"/> this token covers. Allocation-free.</summary>
    public ReadOnlySpan<char> AsSpan() => Source is null ? default : Source.AsSpan(Start, Length);
}

/// <summary>
/// Pure-managed implementation of the Gemma <c>tokenizer.json</c> pipeline (a SentencePiece-style
/// byte-level BPE), replacing the native <c>Tokenizers.HuggingFace</c> dependency.
///
/// The pipeline is:
/// <list type="number">
/// <item><b>Added-token split</b> - the text is split on the 6,415 added/special tokens
/// (e.g. <c>&lt;bos&gt;</c>, <c>&lt;unused0&gt;</c>, runs of newlines), matched leftmost-longest.</item>
/// <item><b>Normalizer</b> - in the gaps between added tokens, every space is replaced with the
/// metaspace marker <c>U+2581</c> (a length-preserving 1:1 substitution, so character offsets are
/// preserved).</item>
/// <item><b>BPE</b> - merge ranks from the merges table are applied with a priority queue; leftover
/// symbols not in the vocabulary fall back to <c>&lt;0xNN&gt;</c> byte tokens.</item>
/// </list>
/// The post-processor (prepend <c>&lt;bos&gt;</c>, append <c>&lt;eos&gt;</c>) is applied by callers
/// that request special tokens.
///
/// Vocabulary and merge tables are keyed by <see cref="UID128"/> (a 128-bit ordinal hash), so
/// runtime lookups never have to materialise a string from a <see cref="ReadOnlySpan{T}"/>.
/// </summary>
internal sealed class GemmaBpe
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _vocabSpan;
    private readonly Dictionary<(string, string), int> _mergeRanks;
    private readonly int[] _byteToId;            // 256 entries: id of "<0xNN>" or -1
    private readonly string[] _idToToken;        // reverse map (id -> surface form) for display
    private readonly AddedTokenTrie _added;
    private readonly int _unkId;

    public int BosId { get; }
    public int EosId { get; }
    public int PadId { get; }

    private const char Metaspace = '▁';

    private GemmaBpe(
        Dictionary<UID128, int> vocab,
        Dictionary<(UID128, UID128), int> mergeRanks,
        int[] byteToId,
        string[] idToToken,
        AddedTokenTrie added,
        int unkId,
        int bosId,
        int eosId,
        int padId)
    {
        _vocab = vocab;
        _vocabSpan = vocab.GetAlternateLookup<ReadOnlySpan<char>>();
        _mergeRanks = mergeRanks;
        _byteToId = byteToId;
        _idToToken = idToToken;
        _added = added;
        _unkId = unkId;
        BosId = bosId;
        EosId = eosId;
        PadId = padId;
    }

    public bool TryGetId(string token, out int id) => _vocab.TryGetValue(token.Hash128(), out id);

    public bool TryGetId(ReadOnlySpan<char> token, out int id) => _vocab.TryGetValue(token.Hash128(), out id);

    /// <summary>Returns the surface string for a vocabulary id (the form used in <c>tokenizer.json</c>,
    /// including the metaspace marker and any <c>&lt;0xNN&gt;</c> byte placeholders).</summary>
    public string IdToToken(int id)
        => (uint)id < (uint)_idToToken.Length ? (_idToToken[id] ?? string.Empty) : string.Empty;

    public bool TryGetId(ReadOnlySpan<char> token, out int id) => _vocabSpan.TryGetValue(token, out id);

    /// <summary>Returns the surface string for a vocabulary id (the form used in <c>tokenizer.json</c>,
    /// including the metaspace marker and any <c>&lt;0xNN&gt;</c> byte placeholders).</summary>
    public string IdToToken(int id)
        => (uint)id < (uint)_idToToken.Length ? (_idToToken[id] ?? string.Empty) : string.Empty;

    public static GemmaBpe FromJson(Stream tokenizerJson)
    {
        using var doc = JsonDocument.Parse(tokenizerJson);
        var root = doc.RootElement;
        var model = root.GetProperty("model");

        // --- vocab ---
        var vocabEl = model.GetProperty("vocab");
        var vocab = new Dictionary<string, int>(vocabEl.GetRawText().Length / 8, StringComparer.Ordinal);
        int maxId = -1;
        foreach (var prop in vocabEl.EnumerateObject())
        {
            int id = prop.Value.GetInt32();
            vocab[prop.Name] = id;
            if (id > maxId) maxId = id;
        }

        // --- merges (ordered; index == rank) ---
        var mergesEl = model.GetProperty("merges");
        var mergeRanks = new Dictionary<(UID128, UID128), int>(mergesEl.GetArrayLength());
        int rank = 0;
        foreach (var m in mergesEl.EnumerateArray())
        {
            // Newer tokenizer.json stores each merge as a two-element array [left, right].
            string left = m[0].GetString()!;
            string right = m[1].GetString()!;
            mergeRanks.TryAdd((left.Hash128(), right.Hash128()), rank);
            rank++;
        }

        // --- byte-fallback table ---
        var byteToId = new int[256];
        for (int b = 0; b < 256; b++)
        {
            var name = "<0x" + b.ToString("X2", CultureInfo.InvariantCulture) + ">";
            byteToId[b] = vocab.TryGetValue(name.Hash128(), out var id) ? id : -1;
        }

        int unkId = vocab.TryGetValue("<unk>".Hash128(), out var u) ? u : 3;

        // --- added tokens ---
        var trie = new AddedTokenTrie();
        int bos = 2, eos = 1, pad = 0;
        var addedList = new List<(string Content, int Id)>();
        if (root.TryGetProperty("added_tokens", out var addedEl))
        {
            foreach (var t in addedEl.EnumerateArray())
            {
                string content = t.GetProperty("content").GetString()!;
                int id = t.GetProperty("id").GetInt32();
                trie.Add(content, id);
                addedList.Add((content, id));
                if (id > maxId) maxId = id;
                switch (content)
                {
                    case "<bos>": bos = id; break;
                    case "<eos>": eos = id; break;
                    case "<pad>": pad = id; break;
                }
            }
        }

        // Reverse map: id -> surface string. Filled from vocab first, then added tokens override
        // any ids that aren't in vocab (defensive; on Gemma's tokenizer.json they coincide).
        var idToToken = new string[maxId + 1];
        foreach (var kv in vocab)
        {
            idToToken[kv.Value] = kv.Key;
        }
        foreach (var (content, id) in addedList)
        {
            if (idToToken[id] is null)
            {
                idToToken[id] = content;
            }
        }

        return new GemmaBpe(vocab, mergeRanks, byteToId, idToToken, trie, unkId, bos, eos, pad);
    }

    /// <summary>Tokenizes <paramref name="text"/>. When <paramref name="addSpecialTokens"/> is true the
    /// result is wrapped with <c>&lt;bos&gt;</c> / <c>&lt;eos&gt;</c> as the post-processor specifies.
    /// Each <see cref="BpeToken"/> in the returned list references <paramref name="text"/> directly -
    /// no per-token substring is allocated.</summary>
    public List<BpeToken> Encode(string text, bool addSpecialTokens)
    {
        var tokens = new List<BpeToken>(text.Length / 3 + 4);
        if (addSpecialTokens)
        {
            tokens.Add(new BpeToken(BosId, text, 0, 0));
        }

        var span = text.AsSpan();
        int pos = 0;
        int n = span.Length;
        while (pos < n)
        {
            // Try to match an added token starting here (leftmost-longest).
            if (_added.TryMatch(span, pos, out int matchLen, out int addedId))
            {
                tokens.Add(new BpeToken(addedId, text, pos, matchLen));
                pos += matchLen;
                continue;
            }

            // Otherwise consume a run of normal text up to the next added-token match (or end),
            // normalize it, and BPE it.
            int runStart = pos;
            pos++;
            while (pos < n && !_added.CouldStartAt(span, pos))
            {
                pos++;
            }
            EncodeSegment(text, span.Slice(runStart, pos - runStart), runStart, tokens);
        }

        if (addSpecialTokens)
        {
            tokens.Add(new BpeToken(EosId, text, n, 0));
        }
        return tokens;
    }

    /// <summary>Normalizes (space -> metaspace) and BPE-encodes <paramref name="segment"/>, which is a
    /// slice of <paramref name="source"/> starting at <paramref name="charOffset"/>. Space -> metaspace
    /// is a 1:1 char substitution so character offsets into <paramref name="source"/> are preserved.</summary>
    private void EncodeSegment(string source, ReadOnlySpan<char> segment, int charOffset, List<BpeToken> output)
    {
        int len = segment.Length;
        if (len == 0)
        {
            return;
        }

        // Fast path: no spaces -> normalization is a no-op, BPE directly over the source slice.
        if (segment.IndexOf(' ') < 0)
        {
            Bpe(source, segment, charOffset, output);
            return;
        }

        // Otherwise rent a buffer for the normalized form. Pooling keeps repeated Encode calls
        // allocation-light. The buffer is only read by Bpe (initial symbols copy 1-2 chars out).
        var rented = ArrayPool<char>.Shared.Rent(len);
        try
        {
            for (int i = 0; i < len; i++)
            {
                char c = segment[i];
                rented[i] = c == ' ' ? Metaspace : c;
            }
            Bpe(source, rented.AsSpan(0, len), charOffset, output);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(rented);
        }
    }

    /// <summary>Runs byte-level BPE over a single normalized segment using a priority-queue merge.
    /// Emitted tokens reference <paramref name="source"/> directly via <paramref name="charOffset"/>.</summary>
    private void Bpe(string source, ReadOnlySpan<char> chars, int charOffset, List<BpeToken> output)
    {
        // Split the segment into Unicode-scalar symbols (so astral characters stay intact).
        // Each BpeSymbol is a (buffer, start, length) slice; merges just grow the left symbol's length.
        var sym = new List<BpeSymbol>(bufferLen);
        int idx = 0;
        while (idx < bufferLen)
        {
            int cl = char.IsHighSurrogate(chars[idx]) && idx + 1 < len && char.IsLowSurrogate(chars[idx + 1]) ? 2 : 1;
            sym.Add(chars.Slice(idx, cl).ToString());
            symStart.Add(idx);
            symLen.Add(cl);
            idx += cl;
        }

        int count = sym.Count;
        if (count == 0)
        {
            return;
        }

        // next/prev form a doubly linked list over the surviving symbols; active marks symbols not
        // yet merged away. Pool the backing arrays so the per-segment overhead is just a rent/return.
        var nextArray = ArrayPool<int>.Shared.Rent(count);
        var prevArray = ArrayPool<int>.Shared.Rent(count);
        var activeArray = ArrayPool<bool>.Shared.Rent(count);
        try
        {
            var next = nextArray.AsSpan(0, count);
            var prev = prevArray.AsSpan(0, count);
            var active = activeArray.AsSpan(0, count);

            for (int i = 0; i < count; i++)
            {
                next[i] = i + 1 < count ? i + 1 : -1;
                prev[i] = i - 1;
                active[i] = true;
            }

            // Min-heap of candidate merges, ordered by (rank, left position) to match BPE
            // tie-breaking: apply the lowest-rank merge first, breaking ties leftmost.
            var heap = new PriorityQueue<PendingMerge, (int Rank, int Left)>();
            for (int i = 0; i < count; i++)
            {
                TryEnqueue(i, next, sym, heap);
            }

            while (heap.TryDequeue(out var pm, out _))
            {
                // Validate the merge is still current: the pair must still be adjacent, both halves
                // active, and neither symbol grown via a different merge since we enqueued.
                if (!active[pm.Left] || next[pm.Left] != pm.Right || !active[pm.Right]
                    || sym[pm.Left].Length != pm.LeftLen
                    || sym[pm.Right].Length != pm.RightLen)
                {
                    continue;
                }

                // Merge right into left: same buffer, same start, extended length.
                var leftSym = sym[pm.Left];
                sym[pm.Left] = new BpeSymbol(leftSym.Buffer, leftSym.Start, leftSym.Length + sym[pm.Right].Length);
                active[pm.Right] = false;
                int rr = next[pm.Right];
                next[pm.Left] = rr;
                if (rr >= 0)
                {
                    prev[rr] = pm.Left;
                }

                // New candidate pairs around the merged symbol.
                TryEnqueue(prev[pm.Left], next, sym, heap);
                TryEnqueue(pm.Left, next, sym, heap);
            }

            // Emit final symbols in order, applying byte-fallback for anything not in the vocabulary.
            for (int i = 0; i >= 0 && i < count; i = next[i])
            {
                if (!active[i])
                {
                    continue;
                }
                var s = sym[i];
                int sStart = charOffset + s.Start;
                int sLen = s.Length;
                var sSpan = s.AsSpan();

                if (_vocab.TryGetValue(sSpan.Hash128(), out int id))
                {
                    output.Add(new BpeToken(id, source, sStart, sLen));
                }
                else
                {
                    EmitByteFallback(source, sSpan, sStart, sLen, output);
                }
            }
        }
        finally
        {
            if (!active[i])
            {
                continue;
            }
            string s = sym[i];
            int sStart = charOffset + symStart[i];
            int sLen = symLen[i];

            if (_vocab.TryGetValue(s, out int id))
            {
                output.Add(new BpeToken(id, source, sStart, sLen));
            }
            else
            {
                EmitByteFallback(source, s, sStart, sLen, output);
            }
        }
    }

    private void EmitByteFallback(string source, string s, int start, int length, List<BpeToken> output)
    {
        // UTF-8 for a single BPE symbol is bounded (a symbol is normally 1-2 UTF-16 code units, so
        // at most 4 UTF-8 bytes). Use a small stack buffer; spill to the heap only for the unusual
        // case of a long unmappable run.
        int maxBytes = Encoding.UTF8.GetMaxByteCount(s.Length);
        Span<byte> buffer = maxBytes <= 256 ? stackalloc byte[256] : new byte[maxBytes];
        int written = Encoding.UTF8.GetBytes(s, buffer);
        for (int i = 0; i < written; i++)
        {
            byte b = buffer[i];
            int id = _byteToId[b];
            output.Add(new BpeToken(id >= 0 ? id : _unkId, source, start, length));
        }

        public ReadOnlySpan<char> AsSpan() => Buffer.AsSpan(Start, Length);
    }

    /// <summary>An entry in the merge priority queue. Lengths are captured at enqueue time and
    /// re-checked on dequeue so a merge that has since been invalidated (because one side absorbed
    /// another symbol first) is silently dropped.</summary>
    private readonly record struct PendingMerge(int Left, int Right, int LeftLen, int RightLen);
}
