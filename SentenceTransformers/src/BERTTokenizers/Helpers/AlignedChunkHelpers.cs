using BERTTokenizers.Base;

namespace SentenceTransformers;

public static class AlignedChunkHelpers
{
    public static TaggedChunk FromOriginal(this TaggedChunkAligned source) => new TaggedChunk(ExtractFromOriginal(source.OriginalText, source.Start, source.ApproximateEnd), source.Tag);

    public static EncodedChunk       FromOriginal(this EncodedChunkAligned       source) => new EncodedChunk(ExtractFromOriginal(source.OriginalText,       source.Start, source.ApproximateEnd), source.Vector);
    public static TaggedEncodedChunk FromOriginal(this TaggedEncodedChunkAligned source) => new TaggedEncodedChunk(ExtractFromOriginal(source.OriginalText, source.Start, source.ApproximateEnd), source.Vector, source.Tag);
    public static string             FromOriginal(this AlignedString             source) => ExtractFromOriginal(source.OriginalText, source.Start, source.ApproximateEnd);
    private static string ExtractFromOriginal(string source, int start, int approximateEnd)
    {
        if (approximateEnd >= source.Length)
        {
            return source.Substring(start);
        }

        if (!char.IsWhiteSpace(source[approximateEnd]))
        {
            var nextSpace = source.IndexOfAny([' ', '\n'], approximateEnd);

            if (nextSpace > 0 && nextSpace - approximateEnd < 10)
            {
                approximateEnd = nextSpace;
            }
        }
        return source.Substring(start, approximateEnd - start);
    }
}