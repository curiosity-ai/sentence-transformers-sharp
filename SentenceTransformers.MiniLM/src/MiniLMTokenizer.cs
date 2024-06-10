/*
 * This file defines the MiniLMTokenizer class, which serves as a tokenizer for the MiniLM model.
 * It inherits from the UncasedTokenizer base class and initializes the tokenizer using the vocabulary file specified by the MiniLM model.
 */
using BERTTokenizers.Base; // Importing the base tokenizer class
using SentenceTransformers; // Importing the SentenceTransformers namespace

namespace SentenceTransformers.MiniLM
{
    // Represents a tokenizer for the MiniLM model.
    public class MiniLMTokenizer : UncasedTokenizer
    {
        // Constructor for MiniLMTokenizer class.
        // Initializes the tokenizer using the vocabulary file specified by the MiniLM model.
        public MiniLMTokenizer() : base(ResourceLoader.OpenResource(typeof(SentenceEncoder).Assembly, "vocab.txt"))
        {
            // No additional logic in the constructor
        }
    }
}