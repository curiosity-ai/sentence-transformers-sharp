using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace SentenceTransformers;

/// <summary>
/// A long-lived, process-wide thread pool dedicated to the CPU-heavy fork/join work that the
/// pure-managed inference kernels do (matmuls, attention, quantization). It is intentionally separate
/// from the .NET <see cref="ThreadPool"/>: those threads are shared with arbitrary library code and are
/// scheduled with a hill-climbing throughput heuristic that interacts badly with the short, fully
/// CPU-bound bursts a forward pass produces. Here the worker count is fixed (by default
/// <see cref="Environment.ProcessorCount"/>), every thread is pinned to
/// <see cref="ThreadPriority.Highest"/> and stays alive for the lifetime of the process, so a parallel
/// fork takes only the wake-up of N already-running threads.
///
/// The <see cref="ForAsync{T}"/> primitive splits <c>[fromInclusive, toExclusive)</c> into one
/// contiguous bucket per worker; the body runs once per bucket and iterates the range itself. The
/// caller-provided state of type <typeparamref name="T"/> is a value type, which lets the body be a
/// <c>static</c> lambda (no closure allocations, no display class) - exactly the shape that a hot
/// matmul loop needs.
///
/// Dispatch is intentionally tight: a single <see cref="SemaphoreSlim.Release(int)"/> wakes N workers
/// in one call, each worker does a short spin before parking so back-to-back forks (the common case
/// for a forward pass) keep the cache hot, and the per-call object graph is two allocations - the job
/// (which is itself the <see cref="System.Threading.Tasks.TaskCompletionSource"/>) plus the body
/// delegate.
/// </summary>
public static class GlobalThreadPool
{
    private static readonly Lock _initLock = new();
    private static volatile bool _initialized;
    private static int _workerCount;
    private static readonly ConcurrentQueue<Action> _queue = new();
    private static readonly SemaphoreSlim _hasWork = new(0);

    /// <summary>The number of worker threads in the pool. Initializes the pool with the defaults
    /// (<see cref="Environment.ProcessorCount"/>, <see cref="ThreadPriority.Highest"/>) on first read
    /// when <see cref="Initialize"/> was not called explicitly.</summary>
    public static int WorkerCount
    {
        get
        {
            EnsureInitialized();
            return _workerCount;
        }
    }

    /// <summary>Configures the pool. Must be called before any work is dispatched (i.e. before the
    /// first <see cref="ForAsync{T}"/> call); subsequent calls are no-ops and return false. If the
    /// pool is used without an explicit call it is auto-initialized with
    /// <see cref="Environment.ProcessorCount"/> threads at <see cref="ThreadPriority.Highest"/>.</summary>
    /// <param name="threadCount">Number of worker threads. Clamped to at least 1.</param>
    /// <param name="priority">OS thread priority for every worker.</param>
    /// <returns>True if this call initialized the pool; false if it was already initialized.</returns>
    public static bool Initialize(int threadCount, ThreadPriority priority)
    {
        if (_initialized)
        {
            return false;
        }
        lock (_initLock)
        {
            if (_initialized)
            {
                return false;
            }
            int n = Math.Max(1, threadCount);
            _workerCount = n;
            _initialized = true;
            for (int i = 0; i < n; i++)
            {
                var t = new Thread(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"SentenceTransformers.GlobalThreadPool[{i}]",
                    Priority = priority,
                };
                t.Start();
            }
            return true;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void EnsureInitialized()
    {
        if (!_initialized)
        {
            Initialize(Environment.ProcessorCount, ThreadPriority.Highest);
        }
    }

    private static void WorkerLoop()
    {
        // Short spin before parking: forward-pass forks come in rapid bursts (multiple matmuls per
        // layer x layers), so a bounded spin keeps cache hot and avoids a kernel wakeup for the
        // common case where the next dispatch arrives within microseconds. The budget is intentionally
        // tiny - SpinWait switches to Sleep(1) past ~10 iterations on .NET, so 30 is enough to cover
        // the gap between back-to-back dispatches without pinning a core.
        const int spinIterations = 30;
        while (true)
        {
            if (_queue.TryDequeue(out var work))
            {
                work();
                continue;
            }
            var spin = new SpinWait();
            while (_queue.IsEmpty && spin.Count < spinIterations)
            {
                spin.SpinOnce(sleep1Threshold: -1);
            }
            if (!_queue.IsEmpty)
            {
                continue;
            }
            _hasWork.Wait();
        }
    }

    /// <summary>
    /// Parallel for-loop over <c>[fromInclusive, toExclusive)</c>: the range is split into
    /// <see cref="WorkerCount"/> (or fewer, for tiny ranges) contiguous buckets and dispatched across
    /// the worker threads. The returned <see cref="Task"/> completes once every bucket has finished;
    /// if any bucket throws, the first observed exception is propagated and the others are dropped.
    ///
    /// The body is invoked once per bucket with the half-open <c>[start, end)</c> range, plus the
    /// caller's <typeparamref name="T"/> state - so callers pass any data the body needs in the state
    /// struct and write the body as a <c>static</c> lambda, avoiding the closure/display-class
    /// allocations the equivalent <see cref="Parallel.ForAsync(int, int, CancellationToken, Func{int, CancellationToken, ValueTask})"/>
    /// per-index lambda would require.
    /// </summary>
    /// <typeparam name="T">Value type carrying any data the body needs. <see cref="ValueTuple"/> is
    /// the usual choice.</typeparam>
    /// <param name="fromInclusive">Lower bound of the iteration range.</param>
    /// <param name="toExclusive">Exclusive upper bound of the iteration range.</param>
    /// <param name="state">Value passed to every bucket invocation of <paramref name="body"/>.</param>
    /// <param name="body">Bucket body. Receives <c>(start, endExclusive, state)</c>.</param>
    /// <param name="ct">Cancellation token. Already-running buckets run to completion; not-yet-started
    /// buckets are skipped and the returned task transitions to <see cref="TaskStatus.Canceled"/>.</param>
    public static Task ForAsync<T>(int fromInclusive, int toExclusive, T state, Action<int, int, T> body, CancellationToken ct = default)
        where T : struct
    {
        if (body is null)
        {
            throw new ArgumentNullException(nameof(body));
        }
        if (ct.IsCancellationRequested)
        {
            return Task.FromCanceled(ct);
        }
        int total = toExclusive - fromInclusive;
        if (total <= 0)
        {
            return Task.CompletedTask;
        }

        EnsureInitialized();

        int workers = Math.Min(_workerCount, total);
        var ctx = new ForContext<T>(workers, state, body, ct, fromInclusive, total);
        // One delegate allocation (method group) reused by every worker; each worker atomically
        // claims its bucket index and computes the contiguous range from it. All buckets but one go
        // to the pool; the caller runs the last bucket inline - the calling thread is about to await
        // this Task anyway, so spending it on real work (instead of one queue-dispatch + thread
        // wake-up) is free, and the caller often gets to return synchronously when its bucket happens
        // to be the last one to finish.
        int dispatched = workers - 1;
        if (dispatched > 0)
        {
            Action runner = ctx.RunNextBucket;
            for (int w = 0; w < dispatched; w++)
            {
                _queue.Enqueue(runner);
            }
            _hasWork.Release(dispatched);
        }
        ctx.RunNextBucket();
        return ctx.Task;
    }

    // The job state and its completion source share one allocation: ForContext _is_ the
    // TaskCompletionSource the caller awaits.
    private sealed class ForContext<T> : TaskCompletionSource where T : struct
    {
        private readonly Action<int, int, T> _body;
        private readonly T _state;
        private readonly CancellationToken _ct;
        private readonly int _from;
        private readonly int _baseSize;
        private readonly int _remainder;
        private int _nextBucket = -1;
        private int _remaining;
        private Exception _error;

        public ForContext(int workers, T state, Action<int, int, T> body, CancellationToken ct, int from, int total)
            : base(TaskCreationOptions.RunContinuationsAsynchronously)
        {
            _remaining = workers;
            _state = state;
            _body = body;
            _ct = ct;
            _from = from;
            _baseSize = total / workers;
            _remainder = total % workers;
        }

        public void RunNextBucket()
        {
            int idx = Interlocked.Increment(ref _nextBucket);
            int start = _from + idx * _baseSize + Math.Min(idx, _remainder);
            int end = start + _baseSize + (idx < _remainder ? 1 : 0);
            try
            {
                if (!_ct.IsCancellationRequested && Volatile.Read(ref _error) is null)
                {
                    _body(start, end, _state);
                }
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref _error, e, null);
            }

            if (Interlocked.Decrement(ref _remaining) == 0)
            {
                var err = _error;
                if (err is not null)
                {
                    SetException(err);
                }
                else if (_ct.IsCancellationRequested)
                {
                    SetCanceled(_ct);
                }
                else
                {
                    SetResult();
                }
            }
        }
    }
}
