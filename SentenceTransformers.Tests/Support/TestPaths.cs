namespace SentenceTransformers.Tests.Support;

internal static class TestPaths
{
    public static string QwenTokenizerJson => Path.Combine(AppContext.BaseDirectory, "Resources", "qwen-tokenizer.json");
    public static string SpeedSampleText => Path.Combine(AppContext.BaseDirectory, "test-data", "text.txt");
    public static string HarrierMediumTokenizerJson => Path.Combine(AppContext.BaseDirectory, "Resources", "harrier-medium-tokenizer.json");
    public static string HarrierSmallTokenizerJson  => Path.Combine(AppContext.BaseDirectory, "Resources", "harrier-small-tokenizer.json");
}
