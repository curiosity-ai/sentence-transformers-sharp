using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;

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
/// The vocabulary is a plain <see cref="Dictionary{TKey, TValue}"/> keyed by the surface string
/// (ordinal comparer); a <see cref="Dictionary{TKey, TValue}.AlternateLookup{TAlternateKey}"/> over
/// <see cref="ReadOnlySpan{T}"/> lets callers probe it without materialising a string.
///
/// The merge table is <b>not</b> keyed by the merge strings. Every merge's left, right and merged
/// (<c>left+right</c>) forms are themselves vocabulary tokens, so a <c>(string, string) -&gt; rank</c>
/// map would retain ~1M string objects duplicating the vocab (two per merge) and hash two strings per
/// probe. Instead each merge is keyed by its two token <b>ids</b> packed into a <see cref="long"/>
/// (<see cref="PackPair"/>), with the rank and the merged token's id stored as the value
/// (<see cref="MergeInfo"/>). The BPE loop therefore works entirely in integer ids: no per-symbol
/// string is materialised and no strings are concatenated as symbols merge.
/// </summary>
internal sealed class GemmaBpe
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _vocabSpan;
    private readonly Dictionary<long, MergeInfo> _merges;
    private readonly int[] _byteToId;            // 256 entries: id of "<0xNN>" or -1
    private readonly string[] _idToToken;        // reverse map (id -> surface form) for display
    private readonly AddedTokenTrie _added;
    private readonly int _unkId;

    /// <summary>The value stored for each merge: its priority <see cref="Rank"/> (lower merges first)
    /// and <see cref="NewId"/>, the vocabulary id of the merged <c>left+right</c> token.</summary>
    private readonly record struct MergeInfo(int Rank, int NewId);

    /// <summary>Packs an ordered pair of token ids into a single <see cref="long"/> merge-table key.
    /// Ids fit in 18 bits (max 262,143), so the two never overlap in the 64-bit key.</summary>
    private static long PackPair(int leftId, int rightId) => ((long)(uint)leftId << 32) | (uint)rightId;

    public int BosId { get; }
    public int EosId { get; }
    public int PadId { get; }

    private const char Metaspace = '▁';

    private GemmaBpe(
        Dictionary<string, int> vocab,
        Dictionary<long, MergeInfo> merges,
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
        _merges = merges;
        _byteToId = byteToId;
        _idToToken = idToToken;
        _added = added;
        _unkId = unkId;
        BosId = bosId;
        EosId = eosId;
        PadId = padId;
    }

    public bool TryGetId(string token, out int id) => _vocab.TryGetValue(token, out id);

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
        var vocabSpan = vocab.GetAlternateLookup<ReadOnlySpan<char>>();

        // --- merges (ordered; index == rank) ---
        // Keyed by packed (leftId, rightId) rather than the surface strings: left, right and the
        // merged left+right form are all vocab tokens, so we store ids and never retain the merge
        // strings (which would duplicate the vocab). The merged string is only needed to resolve its
        // id, which we do through a pooled buffer + the span lookup, so no concat string is allocated.
        var mergesEl = model.GetProperty("merges");
        var merges = new Dictionary<long, MergeInfo>(mergesEl.GetArrayLength());
        char[] concat = new char[64];
        int rank = 0;
        foreach (var m in mergesEl.EnumerateArray())
        {
            // Newer tokenizer.json stores each merge as a two-element array [left, right].
            string left = m[0].GetString()!;
            string right = m[1].GetString()!;
            rank++;
            if (!vocab.TryGetValue(left, out int leftId) || !vocab.TryGetValue(right, out int rightId))
            {
                continue; // a side that isn't a vocab token can never form this merge; skip it
            }
            int need = left.Length + right.Length;
            if (concat.Length < need)
            {
                concat = new char[Math.Max(need, concat.Length * 2)];
            }
            left.CopyTo(concat);
            right.CopyTo(0, concat, left.Length, right.Length);
            if (vocabSpan.TryGetValue(concat.AsSpan(0, need), out int mergedId))
            {
                merges.TryAdd(PackPair(leftId, rightId), new MergeInfo(rank - 1, mergedId));
            }
        }

        // --- byte-fallback table ---
        var byteToId = new int[256];
        for (int b = 0; b < 256; b++)
        {
            var name = "<0x" + b.ToString("X2", CultureInfo.InvariantCulture) + ">";
            byteToId[b] = vocab.TryGetValue(name, out var id) ? id : -1;
        }

        int unkId = vocab.TryGetValue("<unk>", out var u) ? u : 3;

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

        return new GemmaBpe(vocab, merges, byteToId, idToToken, trie, unkId, bos, eos, pad);
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

        // Normalize into a pooled buffer; pooling keeps repeated Encode calls allocation-light. The
        // buffer is only read by Bpe (each initial symbol copies 1-2 chars out of it).
        var rented = ArrayPool<char>.Shared.Rent(len);
        try
        {
            for (int i = 0; i < len; i++)
            {
                char c = segment[i];
                rented[i] = c == ' ' ? Metaspace : c;
            }
            Bpe(source, rented, len, charOffset, output);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(rented);
        }
    }

    /// <summary>Runs byte-level BPE over a single normalized segment using a priority-queue merge.
    /// <paramref name="buffer"/> is the pooled char buffer for the segment (its first
    /// <paramref name="bufferLen"/> chars are the normalized text). Emitted <see cref="BpeToken"/>s
    /// reference <paramref name="source"/> directly via <paramref name="charOffset"/>.</summary>
    private void Bpe(string source, char[] buffer, int bufferLen, int charOffset, List<BpeToken> output)
    {
        // Split the segment into Unicode-scalar symbols (so astral characters stay intact) and resolve
        // each to its vocabulary id up front (-1 when the scalar is not a vocab token; such symbols
        // can never merge and fall through to byte-fallback at emit time). symId/symStart/symLen
        // describe each symbol; next/prev form a doubly linked list; active marks symbols not yet
        // merged away. Working in ids means no per-symbol string is materialised.
        var symId = new List<int>(bufferLen);
        var symStart = new List<int>(bufferLen);   // char offset within the segment
        var symLen = new List<int>(bufferLen);     // char length (1, or 2 for a surrogate pair)
        int idx = 0;
        while (idx < bufferLen)
        {
            int cl = char.IsHighSurrogate(buffer[idx]) && idx + 1 < bufferLen && char.IsLowSurrogate(buffer[idx + 1]) ? 2 : 1;
            symId.Add(_vocabSpan.TryGetValue(buffer.AsSpan(idx, cl), out int id) ? id : -1);
            symStart.Add(idx);
            symLen.Add(cl);
            idx += cl;
        }

        int count = symId.Count;
        if (count == 0)
        {
            return;
        }

        var next = new int[count];
        var prev = new int[count];
        var active = new bool[count];
        for (int i = 0; i < count; i++)
        {
            next[i] = i + 1 < count ? i + 1 : -1;
            prev[i] = i - 1;
            active[i] = true;
        }

        // Min-heap of candidate merges, ordered by (rank, left position) to match BPE tie-breaking:
        // apply the lowest-rank merge first, breaking ties in favour of the leftmost pair.
        var heap = new PriorityQueue<PendingMerge, (int Rank, int Left)>();
        void TryEnqueue(int left)
        {
            if (left < 0)
            {
                return;
            }
            int right = next[left];
            if (right < 0)
            {
                return;
            }
            int leftId = symId[left];
            int rightId = symId[right];
            if (leftId < 0 || rightId < 0)
            {
                return; // an unknown symbol can't be either side of a merge
            }
            if (_merges.TryGetValue(PackPair(leftId, rightId), out var info))
            {
                heap.Enqueue(new PendingMerge(left, right, leftId, rightId, info.NewId), (info.Rank, left));
            }
        }

        for (int i = 0; i < count; i++)
        {
            TryEnqueue(i);
        }

        while (heap.TryDequeue(out var pm, out _))
        {
            // Validate the merge is still current (positions/symbols unchanged since enqueue). A symbol's
            // id changes only when it absorbs a neighbour, so comparing ids detects any staleness.
            if (!active[pm.Left] || next[pm.Left] != pm.Right || !active[pm.Right]
                || symId[pm.Left] != pm.LeftId || symId[pm.Right] != pm.RightId)
            {
                continue;
            }

            // Merge right into left: the merged symbol takes the precomputed merged-token id.
            symId[pm.Left] = pm.NewId;
            symLen[pm.Left] += symLen[pm.Right];
            active[pm.Right] = false;
            int rr = next[pm.Right];
            next[pm.Left] = rr;
            if (rr >= 0)
            {
                prev[rr] = pm.Left;
            }

            // New candidate pairs around the merged symbol.
            TryEnqueue(prev[pm.Left]);
            TryEnqueue(pm.Left);
        }

        // Emit final symbols in order, applying byte-fallback for anything not in the vocabulary.
        for (int i = 0; i >= 0 && i < count; i = next[i])
        {
            if (!active[i])
            {
                continue;
            }
            int id = symId[i];
            int sStart = charOffset + symStart[i];
            int sLen = symLen[i];

            if (id >= 0)
            {
                output.Add(new BpeToken(id, source, sStart, sLen));
            }
            else
            {
                // id < 0 means an unmerged scalar not in the vocab; its chars sit contiguously in the
                // normalized buffer, so byte-fallback reads them straight from there.
                EmitByteFallback(source, buffer.AsSpan(symStart[i], symLen[i]), sStart, sLen, output);
            }
        }
    }

    private void EmitByteFallback(string source, ReadOnlySpan<char> s, int start, int length, List<BpeToken> output)
    {
        // UTF-8 for a single BPE symbol is bounded (a symbol is normally 1-2 UTF-16 code units, so
        // at most 4 UTF-8 bytes). Use a small stack buffer; spill to the heap only for the unusual
        // case of a long unmappable run. The surface form is recovered via IdToToken, so the byte
        // tokens reference the original span rather than allocating "<0xNN>" strings.
        int maxBytes = Encoding.UTF8.GetMaxByteCount(s.Length);
        Span<byte> buffer = maxBytes <= 256 ? stackalloc byte[256] : new byte[maxBytes];
        int written = Encoding.UTF8.GetBytes(s, buffer);
        for (int i = 0; i < written; i++)
        {
            byte b = buffer[i];
            int id = _byteToId[b];
            output.Add(new BpeToken(id >= 0 ? id : _unkId, source, start, length));
        }
    }

    /// <summary>An entry in the merge priority queue. The paired token ids are captured at enqueue time
    /// and re-checked on dequeue so a merge that has since been invalidated (because one side absorbed
    /// another symbol first) is silently dropped. <see cref="NewId"/> is the merged token's id.</summary>
    private readonly record struct PendingMerge(int Left, int Right, int LeftId, int RightId, int NewId);
}
