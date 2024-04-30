using BERTTokenizers.Base;
using MiniLM.Shared;

namespace MiniLM;

public class MiniLMTokenizer : UncasedTokenizer
{
    public MiniLMTokenizer() : base(ResourceLoader.OpenResource(typeof(SentenceEncoder).Assembly, "vocab.txt"))
    {
    }
}