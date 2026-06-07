using System.Runtime.CompilerServices;

namespace SentenceTransformers;

/// <summary>
/// A long-lived, process-wide thread pool dedicated to the CPU-heavy fork/join work that the
/// pure-managed inference kernels do (matmuls, attention, quantization). It is intentionally separate
/// from the .NET <see cref="ThreadPool"/>: those threads are shared with arbitrary library code and are
/// scheduled with a hill-climbing throughput heuristic that interacts badly with the short, fully
/// CPU-bound bursts a forward pass produces. Here the worker count is fixed (by default
/// <see cref="Environment.ProcessorCount"/>), every thread is pinned to
/// <see cref="ThreadPriority.Highest"/> and stays alive for the lifetime of the process.
///
/// The <see cref="ForAsync{T}"/> primitive splits <c>[fromInclusive, toExclusive)</c> into one
/// contiguous bucket per worker; the body runs once per bucket and iterates the range itself. The
/// caller-provided state of type <typeparamref name="T"/> is a value type, which lets the body be a
/// <c>static</c> lambda (no closure allocations, no display class) - exactly the shape that a hot
/// matmul loop needs.
///
/// Dispatch is contention-free in the bucket dimension: every bucket is pre-computed and written
/// directly to a target worker's private mailbox before that worker is signalled. There is no shared
/// work queue, no per-bucket atomic claim, no per-worker scratch on the job - each worker reads
/// <c>{ job, start, end }</c> from its own mailbox and writes nothing the other workers will read.
/// The only shared state the job carries is a single completion counter that each bucket decrements
/// once at the end, plus the first-exception slot. Workers come from / return to a free-worker stack;
/// in the steady state of one producer at a time (a forward pass's sequential awaited matmuls) the
/// stack sees no contention either, and the pool's only remaining lock briefly serializes the (N-1)
/// finishers of a dispatch as they re-enter the free stack.
/// </summary>
public static class GlobalThreadPool
{
    private static readonly Lock _initLock = new();
    private static volatile bool _initialized;
    private static int _workerCount;
    private static Worker[] _workers;

    // Free-worker stack. Guarded by its own lock; only briefly held (push or pop of a few refs).
    private static readonly Lock _freeLock = new();
    private static readonly Stack<Worker> _free = new();

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
            var workers = new Worker[n];
            for (int i = 0; i < n; i++)
            {
                workers[i] = new Worker(i);
            }
            _workers = workers;
            _workerCount = n;
            // Push all workers onto the free stack before starting threads so the first ForAsync
            // call after Initialize finds a full stack.
            for (int i = 0; i < n; i++)
            {
                _free.Push(workers[i]);
            }
            _initialized = true;
            for (int i = 0; i < n; i++)
            {
                workers[i].Start(priority);
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
    ///
    /// All bucket boundaries are computed up front and written to each worker's private mailbox - no
    /// per-bucket atomic claim, no shared work queue. The caller runs the last (smallest) bucket
    /// inline; the dispatched buckets are 0..buckets-2.
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

        int buckets = Math.Min(_workerCount, total);
        int baseSize = total / buckets;
        int remainder = total % buckets;
        // First `remainder` buckets get one extra element; bucket index `buckets-1` is therefore the
        // smallest (when remainder > 0), which is what the caller runs inline.
        int dispatch = buckets - 1;

        var job = new Job<T>(buckets, state, body, ct);

        if (dispatch > 0)
        {
            // Pop up to `dispatch` workers under a brief lock. If the stack is short of workers
            // (concurrent dispatchers, rare in our use case), the overflow buckets run inline below
            // - safe but degrades parallelism for that one dispatch.
            int got;
            Worker[] taken;
            lock (_freeLock)
            {
                got = Math.Min(dispatch, _free.Count);
                taken = got == 0 ? Array.Empty<Worker>() : new Worker[got];
                for (int i = 0; i < got; i++)
                {
                    taken[i] = _free.Pop();
                }
            }

            for (int i = 0; i < got; i++)
            {
                int start = fromInclusive + i * baseSize + Math.Min(i, remainder);
                int end = start + baseSize + (i < remainder ? 1 : 0);
                taken[i].Assign(job, start, end);
            }

            // Buckets we could not dispatch because the free stack was short: run them inline on the
            // caller thread (in addition to the caller's normal inline bucket below). For the common
            // case of one producer at a time this branch never fires.
            for (int i = got; i < dispatch; i++)
            {
                int start = fromInclusive + i * baseSize + Math.Min(i, remainder);
                int end = start + baseSize + (i < remainder ? 1 : 0);
                job.RunBucket(start, end);
            }
        }

        // Caller runs the last (smallest) bucket inline - it is about to await this Task anyway, so
        // spending its thread on real work (instead of one mailbox-write + thread wake-up) is free.
        {
            int i = buckets - 1;
            int start = fromInclusive + i * baseSize + Math.Min(i, remainder);
            int end = start + baseSize + (i < remainder ? 1 : 0);
            job.RunBucket(start, end);
        }

        return job.Task;
    }

    private static void ReturnToFree(Worker w)
    {
        lock (_freeLock)
        {
            _free.Push(w);
        }
    }

    /// <summary>Non-generic façade so a <see cref="Worker"/> can hold an <c>IBucketJob</c> reference
    /// without knowing the closed <c>T</c> at the Job site.</summary>
    private interface IBucketJob
    {
        void RunBucket(int start, int end);
    }

    /// <summary>The job state and its completion source share one allocation: Job _is_ the
    /// TaskCompletionSource the caller awaits. The only fields workers race on are <c>_remaining</c>
    /// (one decrement per bucket; the last one completes the task) and <c>_error</c> (first writer
    /// wins). Bucket boundaries are passed to <see cref="RunBucket"/> directly - there is no shared
    /// bucket counter for workers to contend on.</summary>
    private sealed class Job<T> : TaskCompletionSource, IBucketJob where T : struct
    {
        private readonly Action<int, int, T> _body;
        private readonly T _state;
        private readonly CancellationToken _ct;
        private int _remaining;
        private Exception _error;

        public Job(int buckets, T state, Action<int, int, T> body, CancellationToken ct)
            : base(TaskCreationOptions.RunContinuationsAsynchronously)
        {
            _remaining = buckets;
            _state = state;
            _body = body;
            _ct = ct;
        }

        public void RunBucket(int start, int end)
        {
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

    /// <summary>One pool thread plus its private mailbox. A worker only ever reads its own mailbox
    /// fields, and the dispatcher only ever writes the mailbox of workers it has popped off the free
    /// stack - so once the dispatcher's <see cref="Assign"/> returns the worker races on no
    /// shared mutable state for the duration of the body call.</summary>
    private sealed class Worker
    {
        // Initial 0, no max - we never expect more than one outstanding Release before the worker
        // Waits, but leaving the max unbounded avoids any chance of SemaphoreFullException from a
        // race we missed.
        private readonly SemaphoreSlim _signal = new(0);

        // Mailbox. Written by the dispatcher inside Assign (under happens-before of Release), read
        // by the worker after Wait (under happens-before of Wait's acquire fence). Plain fields are
        // safe under that ordering.
        private IBucketJob _job;
        private int _start;
        private int _end;

        public Worker(int id) { Id = id; }
        public int Id { get; }

        public void Start(ThreadPriority priority)
        {
            var t = new Thread(Loop)
            {
                IsBackground = true,
                Name = $"SentenceTransformers.GlobalThreadPool[{Id}]",
                Priority = priority,
            };
            t.Start();
        }

        public void Assign(IBucketJob job, int start, int end)
        {
            _job = job;
            _start = start;
            _end = end;
            _signal.Release();
        }

        private void Loop()
        {
            while (true)
            {
                _signal.Wait();
                var job = _job;
                int start = _start;
                int end = _end;
                _job = null;

                try
                {
                    job.RunBucket(start, end);
                }
                finally
                {
                    ReturnToFree(this);
                }
            }
        }
    }
}
