namespace BERTTokenizers.Base
{
    public struct TokenizedTokenAligned
    {
        public TokenizedTokenAligned(string token, string original, int start, int approximateEnd)
        {
            Token          = token;
            Original       = original;
            Start          = start;
            ApproximateEnd = approximateEnd;

            if (ApproximateEnd < start)
            {
                throw new Exception();
            }
        }

        public string Token          { get; }
        public string Original       { get; }
        public int    Start          { get; }
        public int    ApproximateEnd { get; }

        public override string ToString()
        {
            return Token;
        }
    }
}