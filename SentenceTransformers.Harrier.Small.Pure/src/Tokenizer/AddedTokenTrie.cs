namespace SentenceTransformers.Harrier.Small.Pure.Tokenizer;

/// <summary>
/// A character trie over the tokenizer's added/special token strings, used to split input text on
/// those tokens before BPE (matching Hugging Face's <c>AddedVocabulary</c> behaviour). Matching is
/// leftmost-longest: at a given position the longest added token that matches wins.
///
/// The added tokens here are control strings such as <c>&lt;bos&gt;</c>, <c>&lt;unused42&gt;</c> and
/// runs of newlines, so they are matched as exact, un-normalized substrings.
///
/// Matching operates over <see cref="ReadOnlySpan{T}"/> so the caller never needs to materialise a
/// substring of the input.
/// </summary>
internal sealed class AddedTokenTrie
{
    private sealed class Node
    {
        public Dictionary<char, Node> Children;
        public int TokenId = -1; // >= 0 marks the end of an added token
    }

    private readonly Node _root = new();

    public void Add(string token, int id)
    {
        if (string.IsNullOrEmpty(token))
        {
            return;
        }
        var node = _root;
        foreach (var c in token)
        {
            node.Children ??= new Dictionary<char, Node>();
            if (!node.Children.TryGetValue(c, out var child))
            {
                child = new Node();
                node.Children[c] = child;
            }
            node = child;
        }
        node.TokenId = id;
    }

    /// <summary>True if any added token could begin at <paramref name="pos"/> (a cheap first-char
    /// gate used to bound the normal-text run scan).</summary>
    public bool CouldStartAt(ReadOnlySpan<char> text, int pos)
        => _root.Children is { } children && children.ContainsKey(text[pos]);

    /// <summary>Attempts the longest added-token match starting at <paramref name="pos"/>.</summary>
    public bool TryMatch(ReadOnlySpan<char> text, int pos, out int matchLength, out int tokenId)
    {
        matchLength = 0;
        tokenId = -1;

        var node = _root;
        int bestLen = 0;
        int bestId = -1;
        int i = pos;
        while (i < text.Length && node.Children is { } children && children.TryGetValue(text[i], out var child))
        {
            node = child;
            i++;
            if (node.TokenId >= 0)
            {
                bestLen = i - pos;
                bestId = node.TokenId;
            }
        }

        if (bestId >= 0)
        {
            matchLength = bestLen;
            tokenId = bestId;
            return true;
        }
        return false;
    }
}
