using BERTTokenizers.Base;
using SentenceTransformers;

namespace SentenceTransformers.Bert.Pure;

/// <summary>
/// The uncased WordPiece tokenizer shared by all-MiniLM-L6-v2 and snowflake-arctic-embed-xs (both use the
/// standard bert-base-uncased 30522-token vocabulary). Embedded so the pure encoder is self-contained.
/// </summary>
public sealed class BertWordPieceTokenizer : UncasedTokenizer
{
    public BertWordPieceTokenizer() : base(ResourceLoader.OpenResource(typeof(BertWordPieceTokenizer).Assembly, "vocab.txt"))
    {
    }
}
