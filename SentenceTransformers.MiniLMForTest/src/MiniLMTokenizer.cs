using BERTTokenizers.Base;
using SentenceTransformers;

namespace SentenceTransformers.MiniLM;

public class MiniLMTokenizer : UncasedTokenizer
{
    public MiniLMTokenizer() : base(ResourceLoader.OpenResource(typeof(SentenceEncoder).Assembly, "vocab.txt"))
    {
    }
}