using BERTTokenizers.Base;
using SentenceTransformers;

namespace SentenceTransformers.ArcticXs;

public class ArcticTokenizer : UncasedTokenizer
{
    public ArcticTokenizer() : base(ResourceLoader.OpenResource(typeof(SentenceEncoder).Assembly, "vocab.txt"))
    {
    }
}