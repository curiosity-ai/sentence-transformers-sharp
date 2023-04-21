using BERTTokenizers.Base;

namespace MiniLM;
public class MiniLMTokenizer : UncasedTokenizer
{
    public MiniLMTokenizer() : base(ResourceLoader.OpenResource(typeof(MiniLMTokenizer).Assembly, "vocab.txt"))
    {
    }
}
