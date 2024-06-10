// This file defines interfaces and data structures related to sentence encoding and chunking.
// It contains the following components:
// - EncodedChunk: Represents a chunk of text along with its corresponding vector representation.
// - TaggedEncodedChunk: Represents a chunk of text along with its corresponding vector representation and a tag.
// - TaggedChunk: Represents a chunk of text along with a tag.
// - ISentenceEncoder: Interface for encoding sentences and chunks of text, providing methods for encoding, chunking, and encoding tagged chunks.
// - TextChunker: Provides utility methods for chunking text.
namespace SentenceTransformers
{
    // Represents a chunk of text along with its corresponding vector representation.
    public record struct EncodedChunk(string Text, float[] Vector);

    // Represents a chunk of text along with its corresponding vector representation and a tag.
    public record struct TaggedEncodedChunk(string Text, float[] Vector, string Tag);

    // Represents a chunk of text along with a tag.
    public record struct TaggedChunk(string Text, string Tag);

    // Interface for encoding sentences and chunks of text.
    public interface ISentenceEncoder
    {
        // Encodes an array of sentences into vectors.
        public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default);

        // Chunks the input text, encodes each chunk, and returns an array of EncodedChunks.
        public EncodedChunk[] ChunkAndEncode(string text, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, CancellationToken cancellationToken = default);

        // Chunks the input text, strips tags, encodes each chunk, and returns an array of TaggedEncodedChunks.
        public TaggedEncodedChunk[] ChunkAndEncodeTagged(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, CancellationToken cancellationToken = default);

        // Chunks the input text into segments of specified length and overlap.
        public static List<string> ChunkText(string text, char separator = ' ', int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue)
        {
            throw new NotImplementedException();
        }
    }

    // Provides utility methods for chunking text.
    public static class TextChunker
    {
        // Splits the input text into chunks of specified length and overlap.
        private static List<string> SplitIntoChunks(IEnumerable<string> splits, char separator, int chunkLength, int chunkOverlap, int maxChunks)
        {
            const int separatorLength = 1;
            var chunks = new List<string>(); // List to hold the resulting chunks of text.
            var currentChunk = new List<string>(); // List to hold the current chunk being constructed.
            int totalLength = 0; // Total length of the current chunk.

            // Iterate through each split of the input text.
            foreach (string split in splits)
            {
                int splitLength = split.Length; // Length of the current split.

                // Check if adding the current split would exceed the chunk length.
                if (totalLength + splitLength + (currentChunk.Count > 0 ? separatorLength : 0) > chunkLength)
                {
                    if (currentChunk.Count > 0)
                    {
                        string chunkText = string.Join(separator, currentChunk);

                        if (!string.IsNullOrWhiteSpace(chunkText))
                        {
                            chunks.Add(chunkText); // Add the completed chunk to the list.
                        }

                        // Remove preceding splits until the overlap condition is satisfied.
                        while (totalLength > chunkOverlap || (totalLength + splitLength + (currentChunk.Count > 0 ? separatorLength : 0) > chunkLength && totalLength > 0))
                        {
                            totalLength -= currentChunk[0].Length + (currentChunk.Count > 1 ? separatorLength : 0);
                            currentChunk.RemoveAt(0);
                        }
                    }
                }

                // Add the current split to the current chunk.
                currentChunk.Add(split);
                totalLength += splitLength + (currentChunk.Count > 1 ? separatorLength : 0); // Update the total length of the chunk.

                if (chunks.Count > maxChunks)
                    return chunks; // Check if the maximum number of chunks is reached.
            }

            // Add the final chunk to the list if it's not empty.
            string finalChunkText = string.Join(separator, currentChunk);
            if (!string.IsNullOrWhiteSpace(finalChunkText))
            {
                chunks.Add(finalChunkText);
            }

            return chunks; // Return the list of chunks.
        }
    }
}