using System.Text.RegularExpressions;
using SentenceTransformers;

namespace SentenceTransformers.Tests.Support;

/// <summary>
/// Re-implementation of the page-marker workflow from <c>SentenceTransformers.TestBase</c>:
/// callers inject <c>⁎N⁑</c> markers into the source text; this helper strips them and recovers
/// the page range as a tag (for example <c>"1"</c> or <c>"1-2"</c>).
/// </summary>
internal static class PageMarker
{
    public const char Start = '⁎';
    public const char End = '⁑';
    private static readonly Regex Pattern = new($"{Start}\\d+{End}", RegexOptions.Compiled);

    public static TaggedChunk StripPageTags(string chunk)
    {
        var pages = new List<int>();
        var cleaned = Pattern.Replace(chunk, match =>
        {
            var page = match.ValueSpan.Slice(1, match.ValueSpan.Length - 2);
            if (int.TryParse(page, out var p)) pages.Add(p);
            return string.Empty;
        });

        if (pages.Count == 0) return new TaggedChunk(cleaned, string.Empty);
        var lo = pages.Min();
        var hi = pages.Max();
        return new TaggedChunk(cleaned, lo == hi ? lo.ToString() : $"{lo}-{hi}");
    }
}
