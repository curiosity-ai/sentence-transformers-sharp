using BERTTokenizers.Helpers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace BERTTokenizers.Base
{
    public abstract class TokenizerBase
    {
        protected readonly List<string>            _vocabulary;
        protected readonly Dictionary<string, int> _vocabularyDict;

        public int MaxWordLength { get; private set; } = 50;
        public int MaxTokens     { get; private set; } = 256;

        public void SetMaxTokens(int maxTokens)
        {
            MaxTokens = maxTokens;
        }

        public void SetMaxWordLength(int maxWordLength)
        {
            MaxWordLength = maxWordLength;
        }

        public TokenizerBase(Stream vocabularyFile)
        {
            _vocabulary = VocabularyReader.ReadFile(vocabularyFile);

            _vocabularyDict = new Dictionary<string, int>();

            for (int i = 0; i < _vocabulary.Count; i++)
            {
                _vocabularyDict[_vocabulary[i]] = i;
            }
        }

        public List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> Encode(params string[] texts)
        {
            var tokenized = Tokenize(MaxTokens, texts);

            if (tokenized.Count == 0)
            {
                return new List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)>();
            }

            int sequenceLength = tokenized.Max(t => Math.Min(MaxTokens, t.Length));

            return tokenized.Select(tokens =>
            {
                var padding = Enumerable.Repeat(0L, sequenceLength - Math.Min(MaxTokens, tokens.Length)).ToList();

                var tokenIndexes   = tokens.Take(MaxTokens).Select(token => (long)token.VocabularyIndex).Concat(padding).ToArray();
                var segmentIndexes = tokens.Take(MaxTokens).Select(token => token.SegmentIndex).Concat(padding).ToArray();
                var inputMask      = tokens.Take(MaxTokens).Select(o => 1L).Concat(padding).ToArray();
                return (tokenIndexes, segmentIndexes, inputMask);
            }).ToList();
        }

        public string IdToToken(int id)
        {
            return _vocabulary[id];
        }

        public List<string> Untokenize(List<string> tokens)
        {
            var currentToken = string.Empty;
            var untokens     = new List<string>();
            tokens.Reverse();

            try
            {

                foreach (var token in tokens)
                {
                    if (token.StartsWith("##"))
                    {
                        currentToken = token.Replace("##", "") + currentToken;
                    }
                    else
                    {
                        currentToken = token + currentToken;
                        untokens.Add(currentToken);
                        currentToken = string.Empty;
                    }
                }
                ;

                untokens.Reverse();
            }
            finally
            {
                tokens.Reverse(); //Need to reverse the list back as we use it later
            }

            return untokens;
        }

        public List<string> Untokenize(List<TokenizedToken> tokens)
        {
            var currentToken = string.Empty;
            var untokens = new List<string>();
            tokens.Reverse();

            try
            {

                foreach (var token in tokens)
                {
                    if (token.Token.StartsWith("##"))
                    {
                        currentToken = token.Token.Replace("##", "") + currentToken;
                    }
                    else
                    {
                        currentToken = (token.Original ?? "") + currentToken;
                        untokens.Add(currentToken);
                        currentToken = string.Empty;
                    }
                }

                untokens.Reverse();
            }
            finally
            {
                tokens.Reverse(); //Need to reverse the list back as we use it later
            }

            return untokens;
        }

        public List<AlignedString> Untokenize(List<TokenizedTokenAligned> tokens)
        {
            var currentToken = string.Empty;
            var untokens = new List<AlignedString>();
            tokens.Reverse();

            try
            {

                foreach (var token in tokens)
                {
                    if (token.Token.StartsWith("##"))
                    {
                        currentToken = token.Token.Replace("##", "") + currentToken;
                    }
                    else
                    {
                        currentToken = (token.Original ?? "") + currentToken;
                        untokens.Add(new AlignedString(currentToken, token.Start, token.Start, token.Start + currentToken.Length));
                        currentToken = string.Empty;
                    }
                }

                untokens.Reverse();
            }
            finally
            {
                tokens.Reverse(); //Need to reverse the list back as we use it later
            }

            return untokens;
        }

        public List<(string Token, int VocabularyIndex, long SegmentIndex)[]> Tokenize(int maxTokens, params string[] texts)
        {
            return texts
               .Select(text =>
                {
                    var tokenAndIndex = new[] { Tokens.Classification }
                       .Concat(TokenizeSentence(Unidecoder.FastUnidecode(RemoveRepeatedSpecialChars(text))).Take(maxTokens))
                       .Concat(new[] { Tokens.Separation })
                       .SelectMany(TokenizeSubwords).Take(maxTokens);
                    var segmentIndexes = SegmentIndex(tokenAndIndex);

                    return tokenAndIndex.Zip(segmentIndexes, (tokenindex, segmentindex)
                        => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToArray();
                })
               .ToList();
        }

        public List<string> TokenizeSimple(string text)
        {
            return TokenizeSentence(Unidecoder.FastUnidecode(RemoveRepeatedSpecialChars(text)))
               .SelectMany(TokenizeSubwords)
               .Select(ti => ti.Token)
               .ToList();
        }

        public List<TokenizedToken> TokenizeRaw(string text)
        {
            return TokenizeSentence(Unidecoder.FastUnidecode(RemoveRepeatedSpecialChars(text)))
               .SelectMany(TokenizeSubwords)
               .Select(ti => new TokenizedToken(ti.Token, ti.Original))
               .ToList();
        }

        public List<TokenizedTokenAligned> TokenizeRawAligned(string text)
        {
            var alignedRemoved = RemoveRepeatedSpecialCharsWithAlignment(text);
            var unidecoderRemoved = Unidecoder.FastUnidecodeWithAlignment(alignedRemoved.text, alignedRemoved.alignment);
            return TokenizeSentenceAligned(unidecoderRemoved.text, unidecoderRemoved.alignment)
               .SelectMany(TokenizeSubwordsAligned)
               .Select(ti => new TokenizedTokenAligned(ti.Token.Value, ti.Original.Value, ti.Token.Start, ti.Token.ApproximateEnd))
               .ToList();
        }


        public static string RemoveRepeatedSpecialChars(string text)
        {
            char last = '\0';
            var sb = new StringBuilder(text.Length);

            foreach (var c in text)
            {
                if (c == last && CharacterClasses.IsSpecialChar(c))
                {
                    continue;
                }
                else
                {
                    last = c;
                    sb.Append(c);
                }
            }
            return sb.ToString();
        }

        public static (string text, List<int> alignment) RemoveRepeatedSpecialCharsWithAlignment(string text)
        {
            char last = '\0';
            var sb = new StringBuilder(text.Length);
            var alignment = new List<int>(text.Length);

            int p = 0;
            foreach (var c in text)
            {
                if (c == last && CharacterClasses.IsSpecialChar(c))
                {
                    p++;
                    continue;
                }
                else
                {
                    last = c;
                    sb.Append(c);
                    alignment.Add(p);
                    p++;
                }
            }
            return (sb.ToString(), alignment);
        }

        private IEnumerable<long> SegmentIndex(IEnumerable<(string token, int index, string original)> tokens)
        {
            var segmentIndex   = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, index, _) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == Tokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        private IEnumerable<(string Token, int VocabularyIndex, string Original)> TokenizeSubwords(string word)
        {
            if (word.Length > MaxWordLength) yield break; //Ignore words that are too long

            if (_vocabularyDict.TryGetValue(word, out var wordIndex))
            {
                yield return (word, wordIndex, word);
                yield break;
            }

            foreach (var inner in TokenizeSubwordsInner(word))
            {
                yield return inner;
            }
        }

        private List<(string token, int index, string Original)> TokenizeSubwordsInner(string word)
        {
            var tokens = new List<(string token, int index, string original)>();
            var remaining = word;

            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                string prefix = null;
                int subwordLength = remaining.Length;

                int stopLimit = remaining.StartsWith("##", StringComparison.Ordinal) ? 2 : 1;

                while (subwordLength >= stopLimit) // was initially 2, which prevents using "character encoding"
                {
                    string subword = remaining.Substring(0, subwordLength);

                    if (!_vocabularyDict.ContainsKey(subword))
                    {
                        subwordLength--;
                        continue;
                    }

                    prefix = subword;
                    break;
                }

                if (string.IsNullOrEmpty(prefix))
                {
                    tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown], TrimStartingHashes(remaining)));

                    return tokens;
                }

                //var regex = new Regex(prefix);
                //remaining = regex.Replace(remaining, "##", 1);


                var remainingAfter = ReplaceFirst(remaining, prefix, "##");

                if (remaining == remainingAfter)
                {
                    tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown], TrimStartingHashes(prefix)));

                    return tokens;
                }
                else
                {
                    remaining = remainingAfter;
                }

                tokens.Add((prefix, _vocabularyDict[prefix], TrimStartingHashes(prefix)));
            }

            if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
            {
                tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown], word));
            }

            return tokens;
        }

        private IEnumerable<(AlignedString Token, int VocabularyIndex, AlignedString Original)> TokenizeSubwordsAligned(AlignedString word)
        {
            if (word.Value.Length > MaxWordLength) yield break; //Ignore words that are too long

            if (_vocabularyDict.TryGetValue(word.Value, out var wordIndex))
            {
                yield return (word, wordIndex, word);
                yield break;
            }

            foreach (var inner in TokenizeSubwordsInnerAligned(word))
            {
                yield return inner;
            }
        }

        private List<(AlignedString token, int index, AlignedString Original)> TokenizeSubwordsInnerAligned(AlignedString word)
        {
            var tokens = new List<(AlignedString token, int index, AlignedString original)>();
            var remaining = word.Value;
            var start = word.Start;

            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                string prefix = null;
                int subwordLength = remaining.Length;

                int stopLimit = remaining.StartsWith("##", StringComparison.Ordinal) ? 2 : 1;

                while (subwordLength >= stopLimit) // was initially 2, which prevents using "character encoding"
                {
                    string subword = remaining.Substring(0, subwordLength);

                    if (!_vocabularyDict.ContainsKey(subword))
                    {
                        subwordLength--;
                        continue;
                    }

                    prefix = subword;
                    break;
                }

                if (string.IsNullOrEmpty(prefix))
                {
                    tokens.Add((new AlignedString(Tokens.Unknown, start, start, start + 1), _vocabularyDict[Tokens.Unknown], new AlignedString(TrimStartingHashes(remaining), start, start, start + 1)));

                    return tokens;
                }

                //var regex = new Regex(prefix);
                //remaining = regex.Replace(remaining, "##", 1);


                var remainingAfter = ReplaceFirst(remaining, prefix, "##");

                if (remaining == remainingAfter)
                {
                    tokens.Add((new AlignedString(Tokens.Unknown, start, start, start + 1), _vocabularyDict[Tokens.Unknown], new AlignedString(TrimStartingHashes(prefix), start, start, start + 1)));

                    return tokens;
                }
                else
                {
                    remaining = remainingAfter;
                }

                tokens.Add((new AlignedString(prefix, start, start, start + prefix.Length), _vocabularyDict[prefix], new AlignedString(TrimStartingHashes(prefix), start, start, start + prefix.Length)));
            }

            if (!string.IsNullOrWhiteSpace(word.Value) && !tokens.Any())
            {
                tokens.Add((new AlignedString(Tokens.Unknown, start, start, start + 1), _vocabularyDict[Tokens.Unknown], word));
            }

            return tokens;
        }

        private static string TrimStartingHashes(string prefix)
        {
            return prefix.StartsWith("##") ? prefix.Substring(2) : prefix;
        }

        private static string ReplaceFirst(string text, string search, string replace)
        {
            int pos = text.IndexOf(search, StringComparison.Ordinal);

            if (pos < 0)
            {
                return text;
            }
            return text.Substring(0, pos) + replace + text.Substring(pos + search.Length);
        }

        public static List<AlignedString> SplitAligned(ReadOnlySpan<char> text, string[] space_delimiters, List<int> alignment)
        {
            var tokens = new List<AlignedString>();
            
            if (text.Length == 0)
            {
                tokens.Add(new AlignedString("", 0, 0, 0));
                return tokens;
            }

            var start = 0;
            while(true)
            {
                var minNextSeparator = int.MaxValue;
                var nextStart = 0;
                foreach(var s in space_delimiters)
                {
                    var nextSeparator = text.Slice(start).IndexOf(s);
                    if (nextSeparator < 0) continue;
                    if (nextSeparator < minNextSeparator)
                    {
                        minNextSeparator = nextSeparator;
                        nextStart = s.Length;
                    }
                }

                if (minNextSeparator < int.MaxValue)
                {
                    tokens.Add(new AlignedString(text.Slice(start, minNextSeparator).ToString(), alignment[start], alignment[start], alignment[start] + minNextSeparator));
                    start += minNextSeparator+nextStart;
                    continue;
                }
                else
                {
                    tokens.Add(new AlignedString(text.Slice(start).ToString(), alignment[start], alignment[start], alignment[start] + (text.Length - start)));
                    break;
                }
            }
            return tokens;
        }


        protected abstract IEnumerable<string> TokenizeSentence(string text);
        protected abstract IEnumerable<AlignedString> TokenizeSentenceAligned(string text, List<int> alignment);
    }
}