using System.Globalization;
using System.Text;
using System.Text.Json;

namespace SentenceTransformers.Harrier.Small.Pure.Tokenizer;

/// <summary>One emitted token: its vocabulary id, surface string, and the [start, end) character span
/// (UTF-16 indices) it covers in the input text.</summary>
internal readonly record struct BpeToken(int Id, string Token, int Start, int End);

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
/// </summary>
internal sealed class GemmaBpe
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _mergeRanks;
    private readonly int[] _byteToId;            // 256 entries: id of "<0xNN>" or -1
    private readonly AddedTokenTrie _added;
    private readonly int _unkId;

    public int BosId { get; }
    public int EosId { get; }
    public int PadId { get; }

    private const char Metaspace = '▁';

    private GemmaBpe(
        Dictionary<string, int> vocab,
        Dictionary<(string, string), int> mergeRanks,
        int[] byteToId,
        AddedTokenTrie added,
        int unkId,
        int bosId,
        int eosId,
        int padId)
    {
        _vocab = vocab;
        _mergeRanks = mergeRanks;
        _byteToId = byteToId;
        _added = added;
        _unkId = unkId;
        BosId = bosId;
        EosId = eosId;
        PadId = padId;
    }

    public bool TryGetId(string token, out int id) => _vocab.TryGetValue(token, out id);

    public static GemmaBpe FromJson(Stream tokenizerJson)
    {
        using var doc = JsonDocument.Parse(tokenizerJson);
        var root = doc.RootElement;
        var model = root.GetProperty("model");

        // --- vocab ---
        var vocabEl = model.GetProperty("vocab");
        var vocab = new Dictionary<string, int>(vocabEl.GetRawText().Length / 8, StringComparer.Ordinal);
        foreach (var prop in vocabEl.EnumerateObject())
        {
            vocab[prop.Name] = prop.Value.GetInt32();
        }

        // --- merges (ordered; index == rank) ---
        var mergesEl = model.GetProperty("merges");
        var mergeRanks = new Dictionary<(string, string), int>(mergesEl.GetArrayLength());
        int rank = 0;
        foreach (var m in mergesEl.EnumerateArray())
        {
            // Newer tokenizer.json stores each merge as a two-element array [left, right].
            string left = m[0].GetString()!;
            string right = m[1].GetString()!;
            mergeRanks.TryAdd((left, right), rank);
            rank++;
        }

        // --- byte-fallback table ---
        var byteToId = new int[256];
        for (int b = 0; b < 256; b++)
        {
            byteToId[b] = vocab.TryGetValue("<0x" + b.ToString("X2", CultureInfo.InvariantCulture) + ">", out var id) ? id : -1;
        }

        int unkId = vocab.TryGetValue("<unk>", out var u) ? u : 3;

        // --- added tokens ---
        var trie = new AddedTokenTrie();
        int bos = 2, eos = 1, pad = 0;
        if (root.TryGetProperty("added_tokens", out var addedEl))
        {
            foreach (var t in addedEl.EnumerateArray())
            {
                string content = t.GetProperty("content").GetString()!;
                int id = t.GetProperty("id").GetInt32();
                trie.Add(content, id);
                switch (content)
                {
                    case "<bos>": bos = id; break;
                    case "<eos>": eos = id; break;
                    case "<pad>": pad = id; break;
                }
            }
        }

        return new GemmaBpe(vocab, mergeRanks, byteToId, trie, unkId, bos, eos, pad);
    }

    /// <summary>Tokenizes <paramref name="text"/>. When <paramref name="addSpecialTokens"/> is true the
    /// result is wrapped with <c>&lt;bos&gt;</c> / <c>&lt;eos&gt;</c> as the post-processor specifies.</summary>
    public List<BpeToken> Encode(string text, bool addSpecialTokens)
    {
        var tokens = new List<BpeToken>(text.Length / 3 + 4);
        if (addSpecialTokens)
        {
            tokens.Add(new BpeToken(BosId, "<bos>", 0, 0));
        }

        int pos = 0;
        int n = text.Length;
        while (pos < n)
        {
            // Try to match an added token starting here (leftmost-longest).
            if (_added.TryMatch(text, pos, out int matchLen, out int addedId))
            {
                tokens.Add(new BpeToken(addedId, text.Substring(pos, matchLen), pos, pos + matchLen));
                pos += matchLen;
                continue;
            }

            // Otherwise consume a run of normal text up to the next added-token match (or end),
            // normalize it, and BPE it.
            int runStart = pos;
            pos++;
            while (pos < n && !_added.CouldStartAt(text, pos))
            {
                pos++;
            }
            EncodeSegment(text, runStart, pos, tokens);
        }

        if (addSpecialTokens)
        {
            tokens.Add(new BpeToken(EosId, "<eos>", n, n));
        }
        return tokens;
    }

    /// <summary>Normalizes (space -> metaspace) and BPE-encodes the text slice [start, end), appending
    /// tokens with character offsets into the original text.</summary>
    private void EncodeSegment(string text, int start, int end, List<BpeToken> output)
    {
        int len = end - start;
        if (len == 0)
        {
            return;
        }

        // Normalize into a buffer; space -> metaspace is a 1:1 char substitution so offsets are preserved.
        var chars = new char[len];
        for (int i = 0; i < len; i++)
        {
            char c = text[start + i];
            chars[i] = c == ' ' ? Metaspace : c;
        }

        Bpe(chars, start, output);
    }

    /// <summary>Runs byte-level BPE over a single normalized segment using a priority-queue merge.</summary>
    private void Bpe(char[] chars, int charOffset, List<BpeToken> output)
    {
        int len = chars.Length;

        // Split the segment into Unicode-scalar symbols (so astral characters stay intact).
        // sym/start/clen describe each symbol; next/prev form a doubly linked list; active marks
        // symbols not yet merged away.
        var sym = new List<string>(len);
        var symStart = new List<int>(len);   // char offset within the segment
        var symLen = new List<int>(len);     // char length (1, or 2 for a surrogate pair)
        int idx = 0;
        while (idx < len)
        {
            int cl = char.IsHighSurrogate(chars[idx]) && idx + 1 < len && char.IsLowSurrogate(chars[idx + 1]) ? 2 : 1;
            sym.Add(new string(chars, idx, cl));
            symStart.Add(idx);
            symLen.Add(cl);
            idx += cl;
        }

        int count = sym.Count;
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
            if (_mergeRanks.TryGetValue((sym[left], sym[right]), out int r))
            {
                heap.Enqueue(new PendingMerge(left, right, sym[left], sym[right]), (r, left));
            }
        }

        for (int i = 0; i < count; i++)
        {
            TryEnqueue(i);
        }

        while (heap.TryDequeue(out var pm, out _))
        {
            // Validate the merge is still current (positions/symbols unchanged since enqueue).
            if (!active[pm.Left] || next[pm.Left] != pm.Right || !active[pm.Right]
                || !string.Equals(sym[pm.Left], pm.LeftSym, StringComparison.Ordinal)
                || !string.Equals(sym[pm.Right], pm.RightSym, StringComparison.Ordinal))
            {
                continue;
            }

            // Merge right into left.
            sym[pm.Left] = pm.LeftSym + pm.RightSym;
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
            string s = sym[i];
            int sStart = charOffset + symStart[i];
            int sEnd = sStart + symLen[i];

            if (_vocab.TryGetValue(s, out int id))
            {
                output.Add(new BpeToken(id, s, sStart, sEnd));
            }
            else
            {
                EmitByteFallback(s, sStart, sEnd, output);
            }
        }
    }

    private void EmitByteFallback(string s, int start, int end, List<BpeToken> output)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        foreach (var b in bytes)
        {
            int id = _byteToId[b];
            if (id >= 0)
            {
                output.Add(new BpeToken(id, "<0x" + b.ToString("X2", CultureInfo.InvariantCulture) + ">", start, end));
            }
            else
            {
                output.Add(new BpeToken(_unkId, "<unk>", start, end));
            }
        }
    }

    private readonly record struct PendingMerge(int Left, int Right, string LeftSym, string RightSym);
}
