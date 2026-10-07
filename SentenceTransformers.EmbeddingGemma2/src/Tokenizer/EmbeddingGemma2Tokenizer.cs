using BERTTokenizers.Base;

namespace SentenceTransformers.EmbeddingGemma2.Tokenizer;

/// <summary>
/// <see cref="TokenizerBase"/> adapter over the pure-managed <see cref="SentencePieceBpe"/> engine, loaded
/// from the SentencePiece model embedded in the EmbeddingGemma 2 <c>.litertlm</c> bundle. Sequences are
/// wrapped as <c>&lt;bos&gt; text &lt;eos&gt;</c>, exactly as the LiteRT-LM embedding engine does.
/// </summary>
public sealed class EmbeddingGemma2Tokenizer : TokenizerBase
{
    private readonly SentencePieceBpe _sp;

    internal SentencePieceBpe SentencePiece => _sp;

    public int BosId { get; }
    public int EosId { get; }
    public int VocabSize => _sp.VocabSize;

    internal EmbeddingGemma2Tokenizer(SentencePieceBpe sp, int bosId, int eosId, int maxTokens)
    {
        _sp = sp;
        BosId = bosId;
        EosId = eosId;
        SetMaxTokens(maxTokens);
        ApproxCharToTokenRatio = 4;
    }

    /// <summary>Loads the tokenizer from a serialized SentencePiece <c>ModelProto</c> (e.g. a <c>tokenizer.model</c>).</summary>
    public static EmbeddingGemma2Tokenizer FromModelProto(byte[] modelProto, int maxTokens = 1024, int bosId = 2, int eosId = 1)
        => new(SentencePieceBpe.FromModelProto(modelProto), bosId, eosId, maxTokens);

    /// <summary>SentencePiece ids for <paramref name="text"/> without special tokens.</summary>
    public int[] EncodeIdsWithoutSpecialTokens(string text)
    {
        var tokens = _sp.Encode(text ?? string.Empty);
        var ids = new int[tokens.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = tokens[i].Id;
        }
        return ids;
    }

    /// <summary>Returns <c>&lt;bos&gt; ids &lt;eos&gt;</c>, keeping at most <see cref="TokenizerBase.MaxTokens"/>
    /// tokens (content is truncated; <c>&lt;eos&gt;</c> is always kept as the last token).</summary>
    public int[] EncodeIds(string text)
    {
        var tokens = _sp.Encode(text ?? string.Empty);
        int content = Math.Min(tokens.Count, Math.Max(0, MaxTokens - 2));
        var ids = new int[content + 2];
        ids[0] = BosId;
        for (int i = 0; i < content; i++)
        {
            ids[i + 1] = tokens[i].Id;
        }
        ids[^1] = EosId;
        return ids;
    }

    public string Decode(ReadOnlySpan<int> ids) => _sp.Decode(ids);

    public override List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> Encode(params string[] texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var result = new List<(long[], long[], long[])>(texts.Length);
        foreach (var text in texts)
        {
            var ids = EncodeIds(text);
            var input = new long[ids.Length];
            var mask = new long[ids.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                input[i] = ids[i];
                mask[i] = 1;
            }
            result.Add((input, new long[ids.Length], mask));
        }
        return result;
    }

    public override string IdToToken(int id) => _sp.IdToPiece(id);

    public override List<string> TokenizeSimple(string text)
    {
        var tokens = _sp.Encode(text ?? string.Empty);
        var result = new List<string>(tokens.Count);
        foreach (var t in tokens)
        {
            result.Add(_sp.IdToPiece(t.Id));
        }
        return result;
    }

    public override List<(string Token, int VocabularyIndex, long SegmentIndex)[]> Tokenize(int maxTokens, params string[] texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var result = new List<(string, int, long)[]>(texts.Length);
        foreach (var text in texts)
        {
            var ids = EncodeIds(text);
            int take = Math.Min(maxTokens, ids.Length);
            var row = new (string, int, long)[take];
            for (int i = 0; i < take; i++)
            {
                row[i] = (_sp.IdToPiece(ids[i]), ids[i], 0L);
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
        var tokens = _sp.Encode(s);
        var result = new List<TokenizedToken>(tokens.Count);
        foreach (var t in tokens)
        {
            if (t.Length <= 0)
            {
                continue;
            }
            result.Add(new TokenizedToken(_sp.IdToPiece(t.Id), s.Substring(t.Start, t.Length)));
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
        var tokens = _sp.Encode(s);
        var result = new List<TokenizedTokenAligned>(tokens.Count);
        foreach (var t in tokens)
        {
            if (t.Length <= 0)
            {
                continue;
            }
            result.Add(new TokenizedTokenAligned(_sp.IdToPiece(t.Id), s.Substring(t.Start, t.Length), t.Start, t.End));
        }
        return result;
    }

    /// <summary>Progress-reporting overload; SentencePiece encodes in one pass, so progress is reported once at the end.</summary>
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
        return new List<AlignedString> { new AlignedString(text, tokens[0].Start, tokens[^1].Start, tokens[^1].ApproximateEnd, originalText) };
    }

    protected override IEnumerable<string> TokenizeSentence(string text)
        => throw new NotSupportedException("EmbeddingGemma2Tokenizer uses SentencePiece BPE; use TokenizeRaw/Encode instead.");

    protected override IEnumerable<AlignedString> TokenizeSentenceAligned(string text, List<int> alignment)
        => throw new NotSupportedException("EmbeddingGemma2Tokenizer uses SentencePiece BPE; use TokenizeRawAligned/Encode instead.");
}
