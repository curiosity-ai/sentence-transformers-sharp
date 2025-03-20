using System.Collections.Concurrent;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using SentenceTransformers;

namespace BERTTokenizers.Base;

public static class Unidecoder
{
    private const           int                              MAX_STACKALLOC_BUFFER_SIZE = 16384;
    private static readonly int                              MaxDecodedCharLength;
    private static          string[][]                       characters;
    private static          ConcurrentDictionary<char, bool> _exceptions = new();

    public static void RegisterException(char c) => _exceptions[c] = true;

    static Unidecoder()
    {
        MaxDecodedCharLength = 0;
        var stream = typeof(Unidecoder).Assembly.GetManifestResourceStream(typeof(Unidecoder).Assembly.GetName().Name + ".Resources." + "unidecoder-decodemap.txt");

        using var reader = new StreamReader(stream, Encoding.UTF8);
        var       lines  = new Dictionary<int, string[]>();
        var       maxidx = -1;

        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            var idx  = int.Parse(line.Substring(0, 3));

            if (idx > maxidx)
            {
                maxidx = idx;
            }
            line = line.Substring(4);
            var pieces = line.Split('\t');

            if (pieces.Length != 256)
            {
                throw new InvalidDataException("Unidecode: malformed data found in embedded resource '" + "unidecoder-decodemap.txt" + "'");
            }

            for (var i = 0; i < pieces.Length; i++)
            {
                var s = pieces[i];
                s = s.Substring(1, s.Length - 2);

                if (s.Length > MaxDecodedCharLength)
                {
                    MaxDecodedCharLength = s.Length;
                }
                pieces[i] = Regex.Unescape(s);
            }

            lines.Add(idx, pieces);
        }
        characters = new string[maxidx + 1][];

        foreach (var pair in lines)
        {
            characters[pair.Key] = pair.Value;
        }

    }

    public static string FastUnidecode(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return "";
        }

        if (input.All(static x => x < 0x80))
        {
            return input;
        }

        var neededBufferSize = input.Length * MaxDecodedCharLength + 1;

        if (neededBufferSize >= MAX_STACKALLOC_BUFFER_SIZE)
        {
            return CompleteUnidecode(input);
        }

        bool       noConversionNeeded = true;
        Span<char> stackBuffer        = stackalloc char[neededBufferSize];
        int        buffIdx            = 0;

        foreach (var c in input)
        {
            if (c < 0x80 || _exceptions.ContainsKey(c))
            {
                stackBuffer[buffIdx++] = c;
                continue;
            }
            noConversionNeeded = false;
            var high = c >> 8;

            if (high >= characters.Length)
            {
                continue;
            }
            var bytes = characters[high];

            if (bytes == null)
            {
                continue;
            }
            var str = bytes[c & 0xff];

            foreach (char ch in str)
            {
                stackBuffer[buffIdx++] = ch;
            }
        }

        if (noConversionNeeded)
        {
            return input;
        }
        return new string(stackBuffer[0..buffIdx]);
    }

    private static string CompleteUnidecode(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return "";
        }

        if (input.All(static x => x < 0x80))
        {
            return input;
        }

        var sb = new StringBuilder(input.Length * 2);

        foreach (var rune in input.EnumerateRunes())
        {
            long c = rune.Value;

            if (c < 0x80)
            {
                sb.Append((char)c);
            }
            else
            {
                var high = c >> 8;

                if (high >= characters.Length)
                {
                    continue;
                }
                var low   = c & 0xff;
                var bytes = characters[high];

                if (bytes != null)
                {
                    sb.Append(bytes[low]);
                }
            }
        }

        return sb.ToString();
    }

    public static (string text, List<int> alignment) FastUnidecodeWithAlignment(string input, List<int> alignment)
    {
        if (string.IsNullOrEmpty(input))
        {
            return ("", alignment);
        }

        if (input.All(static x => x < 0x80))
        {
            return (input, alignment);
        }

        var neededBufferSize = input.Length * MaxDecodedCharLength + 1;

        if (neededBufferSize >= MAX_STACKALLOC_BUFFER_SIZE)
        {
            return CompleteUnidecodeWithAlignment(input, alignment);
        }

        bool       noConversionNeeded = true;
        Span<char> stackBuffer        = stackalloc char[neededBufferSize];

        var newAlignment = TokenizerBase.AlignmentListPool.Rent();

        int buffIdx = 0;

        for (int i = 0; i < input.Length; i++)
        {
            char c = input[i];
            var  p = alignment[i];

            if (c < 0x80 || _exceptions.ContainsKey(c))
            {
                stackBuffer[buffIdx++] = c;
                newAlignment.Add(p);
                continue;
            }
            noConversionNeeded = false;
            var high = c >> 8;

            if (high >= characters.Length)
            {
                continue;
            }
            var bytes = characters[high];

            if (bytes == null)
            {
                continue;
            }
            var str = bytes[c & 0xff];

            foreach (char ch in str)
            {
                stackBuffer[buffIdx++] = ch;
                newAlignment.Add(p);
            }
        }

        if (noConversionNeeded)
        {
            return (input, alignment);
        }

        return (new string(stackBuffer[0..buffIdx]), newAlignment);
    }

    public static (string text, List<int> alignment) CompleteUnidecodeWithAlignment(string input, List<int> alignment)
    {
        if (string.IsNullOrEmpty(input))
        {
            return ("", alignment);
        }

        if (input.All(static x => x < 0x80))
        {
            return (input, alignment);
        }

        var sb = new StringBuilder(input.Length * 2);

        var newAlignment = TokenizerBase.AlignmentListPool.Rent();

        int buffIdx = 0;

        for (int i = 0; i < input.Length; i++)
        {
            char c = input[i];
            var  p = alignment[i];

            if (c < 0x80 || _exceptions.ContainsKey(c))
            {
                sb.Append(c);
                newAlignment.Add(p);
                continue;
            }

            var high = c >> 8;

            if (high >= characters.Length)
            {
                continue;
            }

            var bytes = characters[high];

            if (bytes == null)
            {
                continue;
            }
            var str = bytes[c & 0xff];

            foreach (char ch in str)
            {
                sb.Append(ch);
                newAlignment.Add(p);
            }
        }

        return (sb.ToString(), newAlignment);
    }
}