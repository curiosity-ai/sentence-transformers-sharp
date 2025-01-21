namespace BERTTokenizers.Base
{
    public struct TokenizedToken
    {
        public TokenizedToken(string token, string original)
        {
            Token = token;
            Original = original;
        }

        public string Token { get; set; }
        public string Original { get; set; }
    }
}