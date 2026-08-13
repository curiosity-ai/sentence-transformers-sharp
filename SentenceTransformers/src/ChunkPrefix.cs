using BERTTokenizers.Base;

namespace SentenceTransformers;

/// <summary>
/// A short piece of text put in front of <b>every</b> chunk before it is encoded — a document title
/// in front of each chunk of its body, so a chunk taken from the middle of a document still embeds
/// what the document is about.
///
/// The prefix is tokenized once, when the instance is created, and that is what makes it usable:
/// <list type="bullet">
/// <item>Its token cost is known, so the chunker can be given the room that is actually left
/// (<see cref="EffectiveChunkLength(ChunkPrefix, int)"/>) instead of overflowing the model's context
/// with prefix + a full-size chunk.</item>
/// <item>A long prefix is cropped on a real token boundary to <see cref="MaxTokensFor"/> — never more
/// than half of a chunk and never more than <see cref="MaxTokens"/> tokens — so it can't take over
/// the encoding space that belongs to the chunk itself.</item>
/// </list>
///
/// Pass the instance to the chunking helpers (<c>ChunkTokens</c>, <c>ChunkAndEncodeAsync</c> and their
/// aligned variants). Chunks themselves stay prefix-free — they remain the plain text cut from the
/// source, which is what callers re-chunk to recover a chunk's text — the prefix is only applied to
/// what is handed to the model, via <see cref="Apply(ChunkPrefix, string)"/>.
/// </summary>
public sealed class ChunkPrefix
{
    /// <summary>Hard cap on the prefix, in tokens, however large the chunks are.</summary>
    public const int MaxTokens = 128;

    /// <summary>The largest share of a chunk's token budget the prefix may take (half of it).</summary>
    private const int MaxChunkShareDivisor = 2;

    private ChunkPrefix(string text, int tokenCount, bool wasCropped)
    {
        Text       = text;
        TokenCount = tokenCount;
        WasCropped = wasCropped;
    }

    /// <summary>The prefix as it is prepended to a chunk, separator included.</summary>
    public string Text { get; }

    /// <summary>
    /// How many tokens <see cref="Text"/> costs. Approximate to about a token: the tokenizer may merge
    /// across the boundary between the prefix and the chunk that follows it.
    /// </summary>
    public int TokenCount { get; }

    /// <summary>True when the source prefix was longer than the budget and was cropped to fit it.</summary>
    public bool WasCropped { get; }

    /// <summary>
    /// Tokenizes <paramref name="prefix"/> and crops it to what may lead a chunk of
    /// <paramref name="chunkLength"/> tokens. Returns <c>null</c> for an empty prefix — meaning "no
    /// prefix", which every helper here accepts.
    /// </summary>
    /// <param name="tokenizer">Tokenizer of the encoder the chunks will be encoded with.</param>
    /// <param name="prefix">The text to put in front of each chunk (a title, a heading, …).</param>
    /// <param name="chunkLength">Tokens per chunk the prefix has to share with the chunk's own text.</param>
    /// <param name="separator">Written between the prefix and the chunk. Counts towards the token cost.</param>
    public static ChunkPrefix Create(TokenizerBase tokenizer, string prefix, int chunkLength, string separator = "\n")
    {
        if (tokenizer is null) throw new ArgumentNullException(nameof(tokenizer));

        if (string.IsNullOrWhiteSpace(prefix)) return null;

        prefix    = prefix.Trim();
        separator ??= string.Empty;

        var budget     = MaxTokensFor(chunkLength);
        var tokens     = tokenizer.TokenizeRawAligned(prefix.AsSpan());
        var wasCropped = tokens.Count > budget;

        if (wasCropped)
        {
            // Cut where the last token that fits ends, so the crop never lands inside a token
            var end = Math.Clamp(tokens[budget - 1].ApproximateEnd, 1, prefix.Length);
            prefix  = prefix.Substring(0, end).TrimEnd();
        }

        var text = prefix + separator;

        return new ChunkPrefix(text, tokenizer.TokenizeRaw(text.AsSpan()).Count, wasCropped);
    }

    /// <summary>
    /// The tokens a prefix may take in front of a chunk of <paramref name="chunkLength"/> tokens: at
    /// most half the chunk, and never more than <see cref="MaxTokens"/>.
    /// </summary>
    public static int MaxTokensFor(int chunkLength)
    {
        if (chunkLength <= 0) return MaxTokens;

        return Math.Max(1, Math.Min(MaxTokens, chunkLength / MaxChunkShareDivisor));
    }

    /// <summary>
    /// The tokens left for a chunk's own text once <paramref name="prefix"/> is accounted for. A null
    /// prefix leaves <paramref name="chunkLength"/> untouched.
    /// </summary>
    public static int EffectiveChunkLength(ChunkPrefix prefix, int chunkLength)
    {
        if (prefix is null || chunkLength <= 0) return chunkLength;

        return Math.Max(1, chunkLength - prefix.TokenCount);
    }

    /// <summary>
    /// The text to encode for <paramref name="chunk"/>: the prefix, then the chunk. A null
    /// <paramref name="prefix"/> returns the chunk unchanged.
    /// </summary>
    public static string Apply(ChunkPrefix prefix, string chunk)
    {
        if (prefix is null) return chunk;

        return prefix.Text + chunk;
    }

    /// <inheritdoc/>
    public override string ToString() => Text;
}
