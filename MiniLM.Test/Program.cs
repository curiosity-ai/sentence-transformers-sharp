using System.Diagnostics;
using System.Text;
using MiniLM;


var testSentences = new[] 
{
    "The cat sat on the mat.",
    "The dog sat on the mat.",
    "The kitten sat on the rug",
    "The dog chased the ball.",
    "The deadline for the project is tomorrow at noon.",
    "This project needs to be delivered soon.",
};

var encodedTestSentences = SentenceEncoder.Instance.Encode(testSentences);

var crossSimilarity = new float[testSentences.Length][];

for(int i = 0; i < testSentences.Length; i++)
{
    crossSimilarity[i] = new float[testSentences.Length];
    
    for (int j = 0; j < testSentences.Length; j++)
    {
        crossSimilarity[i][j] = 1f - HNSW.Net.CosineDistance.NonOptimized(encodedTestSentences[i], encodedTestSentences[j]);
    }
}



var sentences = new[]
{
    "The deadline for the project is tomorrow at noon.",
    "To access your account, please enter your username and password.",
    "In order to improve your health, you should exercise regularly and eat a balanced diet.",
    "The conference will take place next week in New York City.",
    "If you have any questions or concerns, please don't hesitate to contact us.",
    "To apply for the job, please submit your resume and cover letter.",
    "The new policy will go into effect starting next month.",
    "In order to succeed, you need to work hard and stay focused.",
    "The company's profits have increased significantly over the past year.",
    "To complete the task, you will need to gather all of the necessary materials.",
};

var sentencesHundred = Enumerable.Range(0, 100).SelectMany(_ => sentences).ToArray();

var results = new List<double[]>();

for (int i = 0; i < 50; i++)
{
    var count = i == 0 ? 1 : i * 10;
    var s = sentencesHundred.Take(count).ToArray();
    count = s.Length;

    var (mean, std) = Profile(5, () =>
    {
        var output = SentenceEncoder.Instance.Encode(s);
    });

    Console.WriteLine($"Senctences {count} Mean {mean:F0}ms Std {std:F0}ms");
    results.Add(new[] { count, mean, std });
}

foreach (var r in results)
{
    Console.WriteLine($"{r[0]},{r[1]:F0},{r[2]:F0}");
}

static (double mean, double avg) Profile(int iterations, Action func)
{
    //Run at highest priority to minimize fluctuations caused by other processes/threads
    Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.High;
    Thread.CurrentThread.Priority = ThreadPriority.Highest;

    // warm up 
    func();

    // clean up
    GC.Collect();
    GC.WaitForPendingFinalizers();
    GC.Collect();

    var times = new double[iterations];

    for (int i = 0; i < iterations; i++)
    {
        var watch = new Stopwatch();
        watch.Start();
        func();
        watch.Stop();
        times[i] = watch.Elapsed.TotalMilliseconds;

        Console.WriteLine($"i: {i} time: {times[i]:F0}ms");

    }
//    Console.Write(description);
//    Console.WriteLine(" Time Elapsed {0} ms", watch.Elapsed.TotalMilliseconds);
    var avg = times.Average();
    var std = Math.Sqrt(times.Average(v => Math.Pow(v - avg, 2)));
    return (avg, std);
}