using BERTTokenizers.Base;

namespace BERTTokenizers
{
    public class BertBaseTokenizer : UncasedTokenizer
    {
        public BertBaseTokenizer() : base("all-MiniLM-L6-v2/vocab.txt")
        {
        }
    }
}
