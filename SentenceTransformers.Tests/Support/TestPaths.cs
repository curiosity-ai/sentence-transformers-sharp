namespace SentenceTransformers.Tests.Support;

internal static class TestPaths
{
    public static string QwenTokenizerJson => Path.Combine(AppContext.BaseDirectory, "Resources", "qwen-tokenizer.json");
    public static string HarrierTokenizerJson => Path.Combine(AppContext.BaseDirectory, "Resources", "harrier-tokenizer.json");
}
