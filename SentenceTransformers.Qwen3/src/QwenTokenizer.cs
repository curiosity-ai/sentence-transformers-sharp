using BERTTokenizers.Base;

// Alias to avoid name collision with our own types
using HFTokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer;
using HFEncoding  = Tokenizers.HuggingFace.Tokenizer.Encoding;

namespace SentenceTransformers.Qwen3
{
    /// <summary>
    /// Thin wrapper around Tokenizers.HuggingFace for tokenizer.json produced by Hugging Face.
    /// Important: includeTypeIds/includeAttentionMask must be true, otherwise Encoding.TypeIds /
    /// Encoding.AttentionMask will be empty.
    /// </summary>
    public sealed class QwenTokenizer : TokenizerBase, IDisposable
    {
        private readonly HFTokenizer _tokenizer;

        // Some tokenizers / FFI wrappers are not guaranteed thread-safe; lock to be safe.
        private readonly object _lock = new();

        public QwenTokenizer(string tokenizerJsonPath, int maxTokens)
        {
            if (string.IsNullOrWhiteSpace(tokenizerJsonPath))
            {
                throw new ArgumentException("tokenizerJsonPath is null/empty", nameof(tokenizerJsonPath));
            }

            if (!File.Exists(tokenizerJsonPath))
            {
                throw new FileNotFoundException("tokenizer.json not found", tokenizerJsonPath);
            }

            _tokenizer = HFTokenizer.FromFile(tokenizerJsonPath);
            SetMaxTokens(maxTokens);
            ApproxCharToTokenRatio = 4; // Byte-level BPE tends to be denser than WordPiece
        }

        public new void SetMaxTokens(int maxTokens)
        {
            if (maxTokens <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokens));
            }
            base.SetMaxTokens(maxTokens);
        }

        /// <summary>
        /// Returns per-sentence token arrays (variable length).
        /// Padding to a uniform length is done in the encoder.
        /// </summary>
        public List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> Encode(string[] sentences, bool addSpecialTokens = true)
        {
            if (sentences is null)
            {
                throw new ArgumentNullException(nameof(sentences));
            }

            var result = new List<(long[], long[], long[])>(sentences.Length);

            foreach (var sentence in sentences)
            {
                var text = sentence ?? string.Empty;

                var enc = EncodeOne(text, addSpecialTokens, includeTypeIds: true, includeTokens: false, includeOffsets: false, includeAttentionMask: true);

                var ids = enc.Ids.Select(x => (long)x).ToArray();

                // Some tokenizers may return empty TypeIds (model may not use them); keep lengths consistent.
                long[] typeIds = enc.TypeIds.Count == ids.Length
                    ? enc.TypeIds.Select(x => (long)x).ToArray()
                    : new long[ids.Length];

                // AttentionMask should be present when includeAttentionMask=true, but be defensive.
                long[] attn = enc.AttentionMask.Count == ids.Length
                    ? enc.AttentionMask.Select(x => (long)x).ToArray()
                    : Enumerable.Repeat(1L, ids.Length).ToArray();

                // Optional truncation (keep arrays aligned)
                if (ids.Length > MaxTokens)
                {
                    ids = ids[..MaxTokens];
                    if (typeIds.Length > MaxTokens)
                    {
                        typeIds = typeIds[..MaxTokens];
                    }
                    if (attn.Length > MaxTokens)
                    {
                        attn = attn[..MaxTokens];
                    }
                }

                result.Add((ids, typeIds, attn));
            }

            return result;
        }

        public override List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> Encode(params string[] texts)
        {
            return Encode(texts, addSpecialTokens: true);
        }

        public override string IdToToken(int id)
        {
            throw new NotSupportedException("QwenTokenizer does not expose IdToToken. Use the HuggingFace tokenizer directly if needed.");
        }

        public override List<string> TokenizeSimple(string text)
        {
            var enc = EncodeOne(text ?? string.Empty, addSpecialTokens: false, includeTypeIds: false, includeTokens: true, includeOffsets: false, includeAttentionMask: false);
            return enc.Tokens.ToList();
        }

        public override List<(string Token, int VocabularyIndex, long SegmentIndex)[]> Tokenize(int maxTokens, params string[] texts)
        {
            if (texts is null)
            {
                throw new ArgumentNullException(nameof(texts));
            }

            var result = new List<(string Token, int VocabularyIndex, long SegmentIndex)[]>(texts.Length);

            foreach (var text in texts)
            {
                var enc = EncodeOne(text ?? string.Empty, addSpecialTokens: true, includeTypeIds: true, includeTokens: true, includeOffsets: false, includeAttentionMask: false);

                int take = Math.Min(maxTokens, enc.Ids.Count);
                var row = new (string Token, int VocabularyIndex, long SegmentIndex)[take];

                for (int i = 0; i < take; i++)
                {
                    row[i] = (enc.Tokens[i], (int)enc.Ids[i], enc.TypeIds.Count > i ? enc.TypeIds[i] : 0L);
                }

                result.Add(row);
            }

            return result;
        }

        public override List<TokenizedToken> TokenizeRaw(ReadOnlySpan<char> text)
        {
            var textString = text.ToString();
            if (textString.Length == 0)
            {
                return new List<TokenizedToken>();
            }

            var enc = EncodeOne(textString, addSpecialTokens: false, includeTypeIds: false, includeTokens: true, includeOffsets: true, includeAttentionMask: false);

            var tokens = enc.Tokens;
            var offsets = enc.Offsets;

            var result = new List<TokenizedToken>(tokens.Count);
            int len = textString.Length;

            for (int i = 0; i < tokens.Count; i++)
            {
                var offset = offsets[i];
                int start = ClampOffset(Convert.ToInt64(offset.Start), len);
                int end = ClampOffset(Convert.ToInt64(offset.End), len);

                if (end <= start)
                {
                    continue; // skip special tokens / empty spans
                }

                var original = textString.Substring(start, end - start);
                result.Add(new TokenizedToken(tokens[i], original));
            }

            return result;
        }

        public override List<TokenizedTokenAligned> TokenizeRawAligned(ReadOnlySpan<char> text)
        {
            var textString = text.ToString();
            if (textString.Length == 0)
            {
                return new List<TokenizedTokenAligned>();
            }

            var enc = EncodeOne(textString, addSpecialTokens: false, includeTypeIds: false, includeTokens: true, includeOffsets: true, includeAttentionMask: false);

            var tokens = enc.Tokens;
            var offsets = enc.Offsets;

            var result = new List<TokenizedTokenAligned>(tokens.Count);
            int len = textString.Length;

            for (int i = 0; i < tokens.Count; i++)
            {
                var offset = offsets[i];
                int start = ClampOffset(Convert.ToInt64(offset.Start), len);
                int end = ClampOffset(Convert.ToInt64(offset.End), len);

                if (end <= start)
                {
                    continue; // skip special tokens / empty spans
                }

                var original = textString.Substring(start, end - start);
                result.Add(new TokenizedTokenAligned(tokens[i], original, start, end));
            }

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
        {
            throw new NotSupportedException("QwenTokenizer uses HuggingFace tokenizers. Use TokenizeRaw/Encode instead.");
        }

        protected override IEnumerable<AlignedString> TokenizeSentenceAligned(string text, List<int> alignment)
        {
            throw new NotSupportedException("QwenTokenizer uses HuggingFace tokenizers. Use TokenizeRawAligned/Encode instead.");
        }

        public void Dispose()
        {
            try { _tokenizer?.Dispose(); } catch { /* swallow */ }
        }

        private HFEncoding EncodeOne(
            string text,
            bool addSpecialTokens,
            bool includeTypeIds,
            bool includeTokens,
            bool includeOffsets,
            bool includeAttentionMask)
        {
            lock (_lock)
            {
                return _tokenizer
                    .Encode(
                        text,
                        addSpecialTokens,
                        includeTypeIds: includeTypeIds,
                        includeTokens: includeTokens,
                        includeWords: false,
                        includeOffsets: includeOffsets,
                        includeSpecialTokensMask: false,
                        includeAttentionMask: includeAttentionMask,
                        includeOverflowing: false)
                    .First();
            }
        }

        private static int ClampOffset(long value, int length)
        {
            if (value < 0)
            {
                return 0;
            }
            if (value > length)
            {
                return length;
            }
            return (int)value;
        }
    }
}
