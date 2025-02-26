using BERTTokenizers.Extensions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BERTTokenizers.Base
{
    public abstract class UncasedTokenizer : TokenizerBase
    {
        private static readonly char[] delimiters = ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray();
        private static readonly string[] space_delimiters = new string[] { " ", "   ", "\r\n" };
        
        protected UncasedTokenizer(Stream vocabularyFile) : base(vocabularyFile) { }

        protected override IEnumerable<string> TokenizeSentence(string text)
        {
            return text.Split(space_delimiters, StringSplitOptions.None).SelectMany(o => o.SplitAndKeep(delimiters)).Select(o => o.ToLower());
        }

        protected override IEnumerable<AlignedString> TokenizeSentenceAligned(string text, List<int> alignment)
        {
            return SplitAligned(text, space_delimiters, alignment).SelectMany(o => StringExtension.SplitAndKeepRaw(o, delimiters)).Select(o => o.ToLower());
        }
    }
}