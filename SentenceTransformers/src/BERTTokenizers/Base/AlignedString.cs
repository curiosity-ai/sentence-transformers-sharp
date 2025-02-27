namespace BERTTokenizers.Base
{
    public struct AlignedString
    {
        public AlignedString(string value, int start, int lastStart, int approximateEnd, string originalText)
        {
            Value = value;
            Start = start;
            LastStart = lastStart;
            ApproximateEnd = approximateEnd;
            OriginalText = originalText;
            if (ApproximateEnd < start)
            {
                throw new Exception();
            }
        }

        public string Value { get; }
        public int Start { get; }
        public int LastStart { get; }
        public int ApproximateEnd { get; }
        public string OriginalText { get; set; }

        public override string ToString()
        {
            return $"{Value} [{Start}-{LastStart}-{ApproximateEnd}]";
        }

        internal AlignedString ToLower()
        {
            return new AlignedString(Value.ToLower(), Start, LastStart, ApproximateEnd, OriginalText);
        }
    }
}

