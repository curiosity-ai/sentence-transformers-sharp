using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace BERTTokenizers.Helpers
{
    internal static class CharacterClasses
    {
        private const           string Currency                          = @"\p{Sc}";
        private const           string Symbols                           = @"\p{So}"; //Symbols like dingbats, but also emoji, see: https://www.compart.com/en/unicode/category/So
        private static readonly Regex  RE_Currency                       = new Regex(Currency);
        private static readonly Regex  RE_IsSymbol                       = new Regex($"^({Symbols})+$");
        private static readonly char[] _whitespacesAndBracketsCharacters = new char[] { ' ', '\n', '\r', '\t', '\v', '\f', '(', ')', '[', ']', '{', '}' }.Union(Enumerable.Range(0, 0x10000).Select(i => ((char)i)).Where(char.IsWhiteSpace)).ToArray();
        private static readonly char[] _currencyCharacters               = Enumerable.Range(0, 0x10000).Select(i => ((char)i)).Where(c => RE_Currency.IsMatch(c.ToString())).ToArray();
        private static readonly char[] _symbolCharacters                 = Enumerable.Range(0, 0x10000).Select(i => ((char)i)).Where(c => RE_IsSymbol.IsMatch(c.ToString())).ToArray();
        private static readonly char[] _hyphenCharacters                 = new char[] { '-', '–', '—', '~' };
        private static readonly char[] _quotesCharacters                 = new char[] { '\'', '"', '”', '“', '`', '‘', '´', '‘', '’', '‚', '„', '»', '«', '「', '」', '『', '』', '（', '）', '〔', '〕', '【', '】', '《', '》', '〈', '〉' };
        private static readonly char[] _sentencePunctuationCharacters    = new char[] { '…', ':', ';', '!', '?', '.' };
        private static readonly char[] _punctuationCharacters            = new char[] { '…', ',', ':', ';', '!', '?', '¿', '¡', '(', ')', '[', ']', '{', '}', '<', '>', '_', '#', '*', '&' };

        private static readonly HashSet<char> _allSpecialChars = _whitespacesAndBracketsCharacters.Concat(_currencyCharacters).Concat(_symbolCharacters).Concat(_hyphenCharacters).Concat(_quotesCharacters).Concat(_sentencePunctuationCharacters).Concat(_punctuationCharacters).ToHashSet();

        public static bool IsSpecialChar(char v) => _allSpecialChars.Contains(v);
    }
}