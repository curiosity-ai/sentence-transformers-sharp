/*
 * This file defines the ArcticTokenizer class, which serves as a tokenizer for the ArcticXs model.
 * The ArcticTokenizer class inherits from the UncasedTokenizer class provided by the BERTTokenizers.Base namespace.
 * It initializes the tokenizer using a vocabulary file named "vocab.txt" loaded from the assembly of the SentenceEncoder type.
 * This tokenizer is designed to tokenize text data according to the specifications of the ArcticXs model.
 */
using BERTTokenizers.Base; // Importing the Base namespace from the BERTTokenizers assembly
using SentenceTransformers; // Importing the SentenceTransformers namespace

namespace SentenceTransformers.ArcticXs
{
    // Defines a tokenizer for the ArcticXs model, inheriting from UncasedTokenizer
    public class ArcticTokenizer : UncasedTokenizer
    {
        // Constructor for ArcticTokenizer
        public ArcticTokenizer() : base(ResourceLoader.OpenResource(typeof(SentenceEncoder).Assembly, "vocab.txt"))
        {
            // Calls the base class constructor with the vocabulary file "vocab.txt"
            // The vocabulary file is loaded using the ResourceLoader.OpenResource method
            // It retrieves the vocab.txt file embedded within the assembly of the SentenceEncoder type
        }
    }
}