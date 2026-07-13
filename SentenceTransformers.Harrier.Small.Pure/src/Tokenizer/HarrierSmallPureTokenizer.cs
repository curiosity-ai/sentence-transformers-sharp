using BERTTokenizers.Base;

namespace SentenceTransformers.Harrier.Small.Pure.Tokenizer;

/// <summary>
/// <see cref="TokenizerBase"/> adapter over the pure-managed <see cref="GemmaBpe"/> engine, providing
/// the same surface the rest of the library expects from the ONNX-backed
/// <c>SentenceTransformers.Harrier.Small</c> tokenizer - but with no native dependency.
/// </summary>
public sealed class HarrierSmallPureTokenizer : TokenizerBase
{
    private readonly GemmaBpe _bpe;
    private readonly object _lock = new();

    internal GemmaBpe Bpe => _bpe;

    internal HarrierSmallPureTokenizer(GemmaBpe bpe, int maxTokens)
    {
        _bpe = bpe;
        SetMaxTokens(maxTokens);
        ApproxCharToTokenRatio = 4; // byte-level BPE is denser than WordPiece
    }

    /// <summary>Loads the tokenizer from a <c>tokenizer.json</c> file on disk.</summary>
    public static HarrierSmallPureTokenizer FromFile(string tokenizerJsonPath, int maxTokens)
    {
        if (string.IsNullOrWhiteSpace(tokenizerJsonPath))
        {
            throw new ArgumentException("tokenizerJsonPath is null/empty", nameof(tokenizerJsonPath));
        }
        if (!File.Exists(tokenizerJsonPath))
        {
            throw new FileNotFoundException("tokenizer.json not found", tokenizerJsonPath);
        }
        using var stream = File.OpenRead(tokenizerJsonPath);
        return new HarrierSmallPureTokenizer(GemmaBpe.FromJson(stream), maxTokens);
    }

    /// <summary>Loads the tokenizer from a <c>tokenizer.json</c> stream.</summary>
    public static HarrierSmallPureTokenizer FromStream(Stream tokenizerJson, int maxTokens)
        => new(GemmaBpe.FromJson(tokenizerJson), maxTokens);

    /// <summary>Tokenizes each sentence to (input ids, token type ids, attention mask), wrapping the
    /// sequence in <c>&lt;bos&gt;</c>/<c>&lt;eos&gt;</c> and truncating to <see cref="TokenizerBase.MaxTokens"/>.
    /// Token type ids are all zero (the model does not use them); the attention mask is all ones.</summary>
    public List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> Encode(string[] sentences, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(sentences);
        var result = new List<(long[], long[], long[])>(sentences.Length);

        foreach (var sentence in sentences)
        {
            var tokens = EncodeTokens(sentence ?? string.Empty, addSpecialTokens);
            int take = Math.Min(tokens.Count, MaxTokens);

            var ids = new long[take];
            var typeIds = new long[take];
            var attn = new long[take];
            for (int i = 0; i < take; i++)
            {
                ids[i] = tokens[i].Id;
                attn[i] = 1L;
            }
            // The model uses last-token (<eos>) pooling, so right-truncation must keep <eos> as the final
            // token rather than lopping it off with the tail content. Mirrors Hugging Face truncation,
            // which preserves post-processor special tokens.
            if (addSpecialTokens && tokens.Count > MaxTokens && take > 0)
            {
                ids[take - 1] = _bpe.EosId;
            }
            result.Add((ids, typeIds, attn));
        }
        return result;
    }

    public override List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> Encode(params string[] texts)
        => Encode(texts, addSpecialTokens: true);

    /// <summary>Returns the integer ids for a sentence (with <c>&lt;bos&gt;</c>/<c>&lt;eos&gt;</c>),
    /// truncated to <see cref="TokenizerBase.MaxTokens"/>. Convenience for the encoder's hot path.</summary>
    public int[] EncodeIds(string text)
    {
        var tokens = EncodeTokens(text ?? string.Empty, addSpecialTokens: true);
        int take = Math.Min(tokens.Count, MaxTokens);
        var ids = new int[take];
        for (int i = 0; i < take; i++)
        {
            ids[i] = tokens[i].Id;
        }
        // Last-token (<eos>) pooling: when the sequence is right-truncated to MaxTokens, the trailing
        // <eos> would otherwise be dropped, leaving the model to pool over an interior content token.
        // Force the final kept token back to <eos> so the pooled embedding stays well-defined.
        if (tokens.Count > MaxTokens && take > 0)
        {
            ids[take - 1] = _bpe.EosId;
        }
        return ids;
    }

    private List<BpeToken> EncodeTokens(string text, bool addSpecialTokens)
    {
        lock (_lock)
        {
            return _bpe.Encode(text, addSpecialTokens);
        }
    }

    public override string IdToToken(int id) => _bpe.IdToToken(id);

    public override List<string> TokenizeSimple(string text)
    {
        var tokens = EncodeTokens(text ?? string.Empty, addSpecialTokens: false);
        var result = new List<string>(tokens.Count);
        foreach (var t in tokens)
        {
            result.Add(_bpe.IdToToken(t.Id));
        }
        return result;
    }

    public override List<(string Token, int VocabularyIndex, long SegmentIndex)[]> Tokenize(int maxTokens, params string[] texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var result = new List<(string, int, long)[]>(texts.Length);
        foreach (var text in texts)
        {
            var tokens = EncodeTokens(text ?? string.Empty, addSpecialTokens: true);
            int take = Math.Min(maxTokens, tokens.Count);
            var row = new (string, int, long)[take];
            for (int i = 0; i < take; i++)
            {
                row[i] = (_bpe.IdToToken(tokens[i].Id), tokens[i].Id, 0L);
            }
            result.Add(row);
        }
        return result;
    }

    public override List<TokenizedToken> TokenizeRaw(ReadOnlySpan<char> text)
    {
        if (text.Length == 0)
        {
            return new List<TokenizedToken>();
        }
        var s = text.ToString();
        var tokens = EncodeTokens(s, addSpecialTokens: false);
        var result = new List<TokenizedToken>(tokens.Count);
        foreach (var t in tokens)
        {
            if (t.Length <= 0)
            {
                continue;
            }
            result.Add(new TokenizedToken(_bpe.IdToToken(t.Id), s.Substring(t.Start, t.Length)));
        }
        return result;
    }

    public override List<TokenizedTokenAligned> TokenizeRawAligned(ReadOnlySpan<char> text)
    {
        if (text.Length == 0)
        {
            return new List<TokenizedTokenAligned>();
        }
        var s = text.ToString();
        var tokens = EncodeTokens(s, addSpecialTokens: false);
        var result = new List<TokenizedTokenAligned>(tokens.Count);
        foreach (var t in tokens)
        {
            if (t.Length <= 0)
            {
                continue;
            }
            result.Add(new TokenizedTokenAligned(_bpe.IdToToken(t.Id), s.Substring(t.Start, t.Length), t.Start, t.End));
        }
        return result;
    }

    /// <summary>
    /// Progress-reporting overload. This tokenizer performs a single opaque Gemma BPE encode, so
    /// progress is reported once at completion (100%); pass <c>null</c> to skip it.
    /// </summary>
    public override List<TokenizedTokenAligned> TokenizeRawAligned(ReadOnlySpan<char> text, IProgress<TokenizeProgress> progress)
    {
        var result = TokenizeRawAligned(text);
        progress?.Report(new TokenizeProgress(text.Length, text.Length, result.Count));
        return result;
    }

    public override List<string> Untokenize(List<TokenizedToken> tokens)
    {
        if (tokens is null || tokens.Count == 0)
        {
            return new List<string>();
        }
        var text = string.Concat(tokens.Select(t => t.Original ?? string.Empty));
        return string.IsNullOrEmpty(text) ? new List<string>() : new List<string> { text };
    }

    public override List<AlignedString> Untokenize(List<TokenizedTokenAligned> tokens, string originalText)
    {
        if (tokens is null || tokens.Count == 0)
        {
            return new List<AlignedString>();
        }
        var text = string.Concat(tokens.Select(t => t.Original ?? string.Empty));
        if (string.IsNullOrEmpty(text))
        {
            return new List<AlignedString>();
        }
        var start = tokens[0].Start;
        var lastStart = tokens[^1].Start;
        var end = tokens[^1].ApproximateEnd;
        return new List<AlignedString> { new AlignedString(text, start, lastStart, end, originalText) };
    }

    protected override IEnumerable<string> TokenizeSentence(string text)
        => throw new NotSupportedException("HarrierSmallPureTokenizer uses Gemma BPE; use TokenizeRaw/Encode instead.");

    protected override IEnumerable<AlignedString> TokenizeSentenceAligned(string text, List<int> alignment)
        => throw new NotSupportedException("HarrierSmallPureTokenizer uses Gemma BPE; use TokenizeRawAligned/Encode instead.");
}
