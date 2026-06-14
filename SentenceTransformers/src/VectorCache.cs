using UID;

namespace SentenceTransformers;

/// <summary>
/// A small, thread-safe least-recently-used (LRU) cache mapping a string's <see cref="UID128"/>
/// hash (produced by <c>text.Hash128()</c>) to the embedding vector most recently encoded for it.
/// Encoders use this to skip re-encoding inputs that were embedded recently. The cache keeps at most
/// <see cref="Capacity"/> entries; inserting beyond that evicts the least recently used one.
/// </summary>
public sealed class VectorCache
{
    private readonly int                                                       _capacity;
    private readonly Dictionary<UID128, LinkedListNode<(UID128 Key, float[] Vector)>> _map;
    private readonly LinkedList<(UID128 Key, float[] Vector)>                   _lru;
    private readonly object                                                    _lock = new();

    /// <summary>Maximum number of cached vectors retained at any time.</summary>
    public int Capacity => _capacity;

    /// <summary>Creates an LRU vector cache.</summary>
    /// <param name="capacity">Maximum number of vectors to retain (defaults to 16).</param>
    public VectorCache(int capacity = 16)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
        _map      = new Dictionary<UID128, LinkedListNode<(UID128, float[])>>(capacity);
        _lru      = new LinkedList<(UID128, float[])>();
    }

    /// <summary>
    /// Looks up the cached vector for <paramref name="key"/>, marking it as most recently used on a hit.
    /// </summary>
    /// <returns><c>true</c> and the cached vector when present; otherwise <c>false</c>.</returns>
    public bool TryGet(UID128 key, out float[] vector)
    {
        lock (_lock)
        {
            if (_map.TryGetValue(key, out var node))
            {
                _lru.Remove(node);
                _lru.AddFirst(node);
                vector = node.Value.Vector;
                return true;
            }
        }

        vector = null;
        return false;
    }

    /// <summary>
    /// Inserts or refreshes the vector for <paramref name="key"/>, evicting the least recently used
    /// entry when the cache is at capacity.
    /// </summary>
    public void Set(UID128 key, float[] vector)
    {
        lock (_lock)
        {
            if (_map.TryGetValue(key, out var existing))
            {
                existing.Value = (key, vector);
                _lru.Remove(existing);
                _lru.AddFirst(existing);
                return;
            }

            if (_map.Count >= _capacity)
            {
                var oldest = _lru.Last;
                if (oldest is object)
                {
                    _lru.RemoveLast();
                    _map.Remove(oldest.Value.Key);
                }
            }

            var node = new LinkedListNode<(UID128, float[])>((key, vector));
            _lru.AddFirst(node);
            _map[key] = node;
        }
    }
}
