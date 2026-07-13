namespace BERTTokenizers.Base
{
    /// <summary>
    /// Progress payload reported by <see cref="TokenizerBase.TokenizeRawAligned(System.ReadOnlySpan{char}, System.IProgress{TokenizeProgress})"/>
    /// while it works through a document. <see cref="CharsProcessed"/> tracks how far into the source
    /// text (in characters) tokenization has reached, so <see cref="Fraction"/> gives a 0..1 estimate
    /// of completion.
    /// </summary>
    public readonly struct TokenizeProgress
    {
        public TokenizeProgress(int charsProcessed, int totalChars, int tokensProduced)
        {
            CharsProcessed = charsProcessed;
            TotalChars     = totalChars;
            TokensProduced = tokensProduced;
        }

        /// <summary>Characters of the source consumed so far.</summary>
        public int CharsProcessed { get; }

        /// <summary>Total characters in the source being tokenized.</summary>
        public int TotalChars { get; }

        /// <summary>Number of aligned tokens produced so far.</summary>
        public int TokensProduced { get; }

        /// <summary>Fraction of the source processed, in the range 0..1 (1 when the source is empty).</summary>
        public double Fraction => TotalChars > 0 ? (double)CharsProcessed / TotalChars : 1.0;

        public override string ToString() => $"{Fraction:P0} ({CharsProcessed}/{TotalChars} chars, {TokensProduced} tokens)";
    }
}
