namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// A low-latency parallel loop for the forward passes, in the spirit of the <c>pthreadpool</c> XNNPACK runs on:
/// persistent background workers spin briefly for the next parallel region before going to sleep, items are
/// claimed with an atomic counter, and the calling thread works too. A region therefore costs a few
/// microseconds instead of a thread-pool dispatch, which matters for the many sub-millisecond operations of
/// a forward pass (e.g. the 16-row GEMMs of the streaming audio encoder).
/// <para>
/// One region runs at a time: a call made while another thread's region is active falls back to
/// <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/>, and calls made from inside a region
/// (nested parallelism) run inline.
/// </para>
/// </summary>
internal static class WorkerPool
{
    private const int SpinIterations = 20_000;   // ~tens of microseconds before a worker sleeps

    private sealed class Job
    {
        public Action<int> Body;
        public int Count;
        public int MaxWorkers;
        public int Next;
        public int Completed;
        public Exception Error;
    }

    private static readonly object Gate = new();
    private static readonly int WorkerCount = Math.Max(0, Environment.ProcessorCount - 1);
    private static Thread[] _threads;
    private static Job _job;
    private static long _generation;
    private static int _busy;

    [ThreadStatic]
    private static bool _inside;

    /// <summary>Runs <paramref name="body"/> for <c>0 ≤ i &lt; count</c> on up to <paramref name="po"/>'s degree of
    /// parallelism threads (the caller included) and returns when all items are done.</summary>
    public static void For(int count, ParallelOptions po, Action<int> body)
    {
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }
        For(count, dop, body);
    }

    /// <inheritdoc cref="For(int, ParallelOptions, Action{int})"/>
    public static void For(int count, int dop, Action<int> body)
    {
        if (count <= 0)
        {
            return;
        }
        if (dop <= 1 || count == 1 || _inside || WorkerCount == 0)
        {
            for (int i = 0; i < count; i++)
            {
                body(i);
            }
            return;
        }
        if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
        {
            Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = dop }, body);
            return;
        }
        try
        {
            EnsureStarted();
            var job = new Job { Body = body, Count = count, MaxWorkers = Math.Min(dop - 1, WorkerCount) };
            lock (Gate)
            {
                Volatile.Write(ref _job, job);
                _generation++;
                Monitor.PulseAll(Gate);
            }
            _inside = true;
            try
            {
                Work(job);
            }
            finally
            {
                _inside = false;
            }
            var spin = new SpinWait();
            while (Volatile.Read(ref job.Completed) < count)
            {
                spin.SpinOnce(sleep1Threshold: -1);
            }
            Volatile.Write(ref _job, null);
            if (job.Error is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(job.Error).Throw();
            }
        }
        finally
        {
            Volatile.Write(ref _busy, 0);
        }
    }

    private static void Work(Job job)
    {
        int i;
        while ((i = Interlocked.Increment(ref job.Next) - 1) < job.Count)
        {
            try
            {
                if (job.Error is null)
                {
                    job.Body(i);
                }
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref job.Error, e, null);
            }
            finally
            {
                Interlocked.Increment(ref job.Completed);
            }
        }
    }

    private static void EnsureStarted()
    {
        if (Volatile.Read(ref _threads) is not null)
        {
            return;
        }
        lock (Gate)
        {
            if (_threads is not null)
            {
                return;
            }
            var threads = new Thread[WorkerCount];
            for (int w = 0; w < threads.Length; w++)
            {
                int index = w;
                threads[w] = new Thread(() => WorkerLoop(index))
                {
                    IsBackground = true,
                    Name = $"EmbeddingGemma2 worker {index}",
                };
                threads[w].Start();
            }
            Volatile.Write(ref _threads, threads);
        }
    }

    private static void WorkerLoop(int index)
    {
        _inside = true;
        long seen = 0;
        while (true)
        {
            int spins = 0;
            while (Volatile.Read(ref _generation) == seen && spins < SpinIterations)
            {
                Thread.SpinWait(1);
                spins++;
            }
            if (Volatile.Read(ref _generation) == seen)
            {
                lock (Gate)
                {
                    while (_generation == seen)
                    {
                        Monitor.Wait(Gate);
                    }
                }
            }
            seen = Volatile.Read(ref _generation);
            var job = Volatile.Read(ref _job);
            if (job is not null && index < job.MaxWorkers)
            {
                Work(job);
            }
        }
    }
}
