using BERTTokenizers.Extensions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BERTTokenizers.Base
{
    public abstract class CasedTokenizer : TokenizerBase
    {
        private static readonly char[] delimiters = ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray();
        private static readonly string[] space_delimiters = new string[] { " ", "   ", "\r\n" };

        protected CasedTokenizer(Stream vocabularyFile) : base(vocabularyFile) { }
        
     
        protected override IEnumerable<string> TokenizeSentence(string text)
        {
            return text.Split(space_delimiters, StringSplitOptions.None).SelectMany(o => o.SplitAndKeep(delimiters));
        }

        protected override IEnumerable<AlignedString> TokenizeSentenceAligned(string text, List<int> alignment)
        {
            return SplitAligned(text, space_delimiters, alignment).SelectMany(o => StringExtension.SplitAndKeepRaw(o, delimiters));
        }
    }
}