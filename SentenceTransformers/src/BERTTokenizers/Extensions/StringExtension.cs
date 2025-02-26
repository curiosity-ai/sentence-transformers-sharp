using BERTTokenizers.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BERTTokenizers.Extensions
{
    static class StringExtension
    {
        public static IEnumerable<string> SplitAndKeep(this string inputString, params char[] delimiters)
        {
            int start = 0, index;

            while ((index = inputString.IndexOfAny(delimiters, start)) != -1)
            {
                if (index - start > 0)
                    yield return inputString.Substring(start, index - start);

                yield return inputString.Substring(index, 1);

                start = index + 1;
            }

            if (start < inputString.Length)
            {
                yield return inputString.Substring(start);
            }
        }


        public static IEnumerable<AlignedString> SplitAndKeepRaw(AlignedString inputToken, params char[] delimiters)
        {
            int start = 0, index;

            while ((index = inputToken.Value.IndexOfAny(delimiters, start)) != -1)
            {
                if (index - start > 0)
                    yield return new AlignedString(inputToken.Value.Substring(start, index - start), start + inputToken.Start, start + inputToken.Start, start + inputToken.Start + (index - start));

                yield return new AlignedString(inputToken.Value.Substring(index, 1), index + inputToken.Start, index + start + inputToken.Start, index + start + inputToken.Start + 1);

                start = index + 1;
            }

            if (start < inputToken.Value.Length)
            {
                yield return new AlignedString(inputToken.Value.Substring(start), start + inputToken.Start, start + inputToken.Start, start + inputToken.Start + (inputToken.Value.Length - start));
            }
        }

    }
}