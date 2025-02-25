using BERTTokenizers.Base;

namespace SentenceTransformers;

public record struct EncodedChunk(string       Text, float[] Vector);
public record struct TaggedEncodedChunk(string Text, float[] Vector, string Tag);
public record struct TaggedChunk(string        Text, string  Tag);
public interface ISentenceEncoder
{
    public int MaxChunkLength { get; }

    public TokenizerBase Tokenizer { get; }

    public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default);

    public EncodedChunk[] ChunkAndEncode(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MaxChunkLength)
        {
            chunkLength = MaxChunkLength;
        }

        if (chunkOverlap < 0 || chunkOverlap > chunkLength)
        {
            chunkOverlap = chunkLength / 5;
        }

        var chunks = ChunkTokens(text, chunkLength, chunkOverlap, maxChunks);

        var encodedChunks = new EncodedChunk[chunks.Count];

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i];
                    var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new EncodedChunk(oneChunk[0], oneVector[0]);
                }
            }
            else
            {
                var vectors = Encode(chunks.ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new EncodedChunk(chunks[i], vectors[i]);
                }
            }
        }
        catch (Exception E)
        {
            if ((E is OperationCanceledException || cancellationToken.IsCancellationRequested) && keepResultsOnCancellation)
            {
                return encodedChunks.Where(c => c != null).ToArray();
            }
            else
            {
                throw;
            }
        }

        return encodedChunks;
    }

    public TaggedEncodedChunk[] ChunkAndEncodeTagged(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MaxChunkLength)
        {
            chunkLength = MaxChunkLength;
        }

        if (chunkOverlap < 0 || chunkOverlap > chunkLength)
        {
            chunkOverlap = chunkLength / 5;
        }

        var chunks = ChunkTokens(text, chunkLength, chunkOverlap, maxChunks: maxChunks)
                       .Select(chunk => stripTags(chunk))
                       .ToArray();

        var encodedChunks = new TaggedEncodedChunk[chunks.Length];

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i].Text;
                    var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, oneVector[0], chunks[i].Tag);
                }
            }
            else
            {
                var vectors = Encode(chunks.Select(c => c.Text).ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, vectors[i], chunks[i].Tag);
                }
            }
        }
        catch (Exception E)
        {
            if ((E is OperationCanceledException || cancellationToken.IsCancellationRequested) && keepResultsOnCancellation)
            {
                return encodedChunks.Where(c => c != null).ToArray();
            }
            else
            {
                throw;
            }
        }

        return encodedChunks;
    }

    public List<string> ChunkTokens(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue)
    {
        return MergeTokenSplits(Tokenizer.TokenizeRaw(text), chunkLength, chunkOverlap, maxChunks);
    }

    private List<string> MergeTokenSplits(IEnumerable<TokenizedToken> splits, int chunkLength, int chunkOverlap, int maxChunks)
    {
        const int separatorLength = 1;
        var       docs            = new List<string>();
        var       currentDoc      = new List<TokenizedToken>();

        foreach (var d in splits)
        {
            if (currentDoc.Count + 1 > chunkLength)
            {
                if (currentDoc.Count > 0)
                {
                    if (currentDoc.Any(c => !string.IsNullOrWhiteSpace(c.Original)))
                    {
                        var untokenized = string.Join(' ', Tokenizer.Untokenize(currentDoc));
                        docs.Add(untokenized);
                    }

                    while (currentDoc.Count > chunkOverlap || (currentDoc.Count + 1 > chunkLength && currentDoc.Count > 0))
                    {
                        currentDoc.RemoveAt(0);
                    }
                }
            }
            currentDoc.Add(d);

            if (docs.Count > maxChunks)
            {
                return docs;
            }
        }

        string final_doc = string.Join(' ', Tokenizer.Untokenize(currentDoc));

        if (!string.IsNullOrWhiteSpace(final_doc))
        {
            docs.Add(final_doc);
        }

        return docs;
    }

    public static List<string> ChunkString(string text, char separator = ' ', int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue)
    {
        return MergeStringSplits(text.Split(new char[] { '\n', '\r', ' ' }, StringSplitOptions.RemoveEmptyEntries), separator, chunkLength, chunkOverlap, maxChunks);
    }

    private static List<string> MergeStringSplits(IEnumerable<string> splits, char separator, int chunkLength, int chunkOverlap, int maxChunks)
    {
        const int separatorLength = 1;
        var       docs            = new List<string>();
        var       currentDoc      = new List<string>();
        int       total           = 0;

        foreach (string d in splits)
        {
            int len = d.Length;

            if (total + len + (currentDoc.Count > 0 ? separatorLength : 0) > chunkLength)
            {
                if (currentDoc.Count > 0)
                {
                    string doc = string.Join(separator, currentDoc);

                    if (!string.IsNullOrWhiteSpace(doc))
                    {
                        docs.Add(doc);
                    }

                    while (total > chunkOverlap || (total + len + (currentDoc.Count > 0 ? separatorLength : 0) > chunkLength && total > 0))
                    {
                        total -= currentDoc[0].Length + (currentDoc.Count > 1 ? separatorLength : 0);
                        currentDoc.RemoveAt(0);
                    }
                }
            }
            currentDoc.Add(d);
            total += len + (currentDoc.Count > 1 ? separatorLength : 0);

            if (docs.Count > maxChunks) return docs;
        }

        string final_doc = string.Join(separator, currentDoc);

        if (!string.IsNullOrWhiteSpace(final_doc))
        {
            docs.Add(final_doc);
        }

        return docs;
    }
    
    public List<string> SplitOriginalBasedOnChunks(string textToChunk, List<string> chunks, int chunkLength = 500, int chunkOverlap = 100)
    {
        var result = new List<string>();
        var unidecoded = Unidecoder.FastUnidecode(TokenizerBase.RemoveRepeatedSpecialChars(textToChunk)); //Apply same transformation as during tokenization

        var scanRange = textToChunk.Length - unidecoded.Length; //We scan for distance keeping based on the max difference between the original and cleaned string

        ReadOnlySpan<char> textToSearch = textToChunk;

        var overlapFactor = (double)chunkOverlap / chunkLength;
        bool isFirst = true;
        foreach (var chunk in chunks)
        {
            var bestMatch =  FindBestMatchOverlap(textToSearch.Slice(0, Math.Min(textToSearch.Length, chunk.Length + scanRange * 2)), chunk, isFirst);
            isFirst = false;
            textToSearch = textToSearch.Slice((int)(bestMatch.Length * (1 - overlapFactor)));
            result.Add(bestMatch.ToString());
        }

        return result;
    }

    public static ReadOnlySpan<char> FindBestMatchOverlap(ReadOnlySpan<char> text, ReadOnlySpan<char> needle, bool isFirst)
    {
        double bestDistance = double.MaxValue;
        ReadOnlySpan<char> bestMatch = null;
        
        var delta = text.Length - needle.Length;

        for (int i = 0; i <= delta; i++)
        {
            var window = text.Slice(isFirst ? 0 : i, needle.Length);
            var distance = LevenshteinDistance(window, needle);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestMatch = window;
            }
        }

        return bestMatch;
    }
    public static int LevenshteinDistance(ReadOnlySpan<char> s1, ReadOnlySpan<char> s2)
    {
        int lenS1 = s1.Length;
        int lenS2 = s2.Length;
        int[,] matrix = new int[lenS1 + 1, lenS2 + 1];

        for (int i = 0; i <= lenS1; matrix[i, 0] = i++) ;
        for (int j = 0; j <= lenS2; matrix[0, j] = j++) ;

        for (int i = 1; i <= lenS1; i++)
        {
            for (int j = 1; j <= lenS2; j++)
            {
                int cost = FuzzyEquals(s1[i - 1],s2[j - 1]) ? 0 : 1;
                matrix[i, j] = Math.Min(Math.Min(
                    matrix[i - 1, j] + 1,  // deletion
                    matrix[i, j - 1] + 1), // insertion
                    matrix[i - 1, j - 1] + cost); // substitution
            }
        }
        return matrix[lenS1, lenS2];
    }

    private static bool FuzzyEquals(char a, char b)
    {
        a = char.ToLowerInvariant(a);
        b = char.ToLowerInvariant(b);
        if (a == b) return true;
        if (Unidecoder.FastUnidecode(a.ToString()) == Unidecoder.FastUnidecode(b.ToString())) return true;
        return false;
    }



    //private const double defaultGapCost = 2.0;
    //private const double defaultMinCost = 0;
    //private const double defaultMismatchScore = 0.0;
    //private const double defaultPerfectMatchScore = 1.0;
    //private static double GetSimilarity(string firstWord, string secondWord)
    //{
    //    if ((firstWord == null) || (secondWord == null))
    //        return 0.0;

    //    double unnormalizedSimilarity = GetUnnormalizedSimilarity(firstWord, secondWord);
    //    double num2 = Math.Max(firstWord.Length, secondWord.Length);
    //    double num3 = num2;
    //    num2 *= defaultGapCost;
    //    num3 *= defaultMinCost;

    //    if (num3 < 0.0)
    //    {
    //        num2 -= num3;
    //        unnormalizedSimilarity -= num3;
    //    }
    //    if (num2 == 0.0)
    //        return 1.0;

    //    return (1.0 - (unnormalizedSimilarity / num2));
    //}

    //private static double GetUnnormalizedSimilarity(string firstWord, string secondWord)
    //{
    //    if ((firstWord == null) || (secondWord == null))
    //        return 0.0;

    //    int length = firstWord.Length;
    //    int index = secondWord.Length;
    //    if (length == 0)
    //        return (double)index;

    //    if (index == 0)
    //        return (double)length;

    //    double[][] numArray = new double[length + 1][];
    //    for (int i = 0; i < (length + 1); i++)
    //    {
    //        numArray[i] = new double[index + 1];
    //    }
    //    for (int j = 0; j <= length; j++)
    //    {
    //        numArray[j][0] = j;
    //    }
    //    for (int k = 0; k <= index; k++)
    //    {
    //        numArray[0][k] = k;
    //    }
    //    for (int m = 1; m <= length; m++)
    //    {
    //        for (int n = 1; n <= index; n++)
    //        {
    //            double num8 = GetCost(firstWord, m - 1, secondWord, n - 1);
    //            numArray[m][n] = Math.Min(Math.Min((double)(numArray[m - 1][n] + defaultGapCost), (double)(numArray[m][n - 1] + defaultGapCost)), (double)(numArray[m - 1][n - 1] + num8));
    //        }
    //    }
    //    return numArray[length][index];
    //}

    //private static double GetCost(string firstWord, int firstWordIndex, string secondWord, int secondWordIndex)
    //{
    //    if ((firstWord != null) && (secondWord != null))
    //        return ((firstWord[firstWordIndex] != secondWord[secondWordIndex]) ? ((double)1) : ((double)0));

    //    return 0.0;
    //}

}