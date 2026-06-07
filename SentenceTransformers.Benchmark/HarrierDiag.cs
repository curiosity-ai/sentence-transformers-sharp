using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;

public static class HarrierDiag
{
    public static async Task RunAsync()
    {
        using var enc = await SentenceEncoder.CreateAsync();

        var queries = new[] { "how much protein should a female eat", "summit define" };
        var documents = new[]
        {
            "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        };

        var q = await enc.EncodeQueriesAsync(queries, SentenceEncoder.Prompts.WebSearchQuery);
        var d = await enc.EncodeAsync(documents);

        float[,] expected = { { 64.57f, 25.45f }, { 29.97f, 67.41f } };
        for (int i = 0; i < q.Length; i++)
        {
            for (int j = 0; j < d.Length; j++)
            {
                float dot = 0;
                for (int k = 0; k < q[i].Length; k++) dot += q[i][k] * d[j][k];
                float score = dot * 100f;
                Console.WriteLine($"score[{i},{j}]={score:F2} expected {expected[i, j]:F2} diff={score - expected[i,j]:+0.00;-0.00}");
            }
        }

        for (int i = 0; i < q.Length; i++)
        {
            float n = 0;
            for (int k = 0; k < q[i].Length; k++) n += q[i][k] * q[i][k];
            Console.WriteLine($"|q[{i}]|^2 = {n:F6}  first5=[{q[i][0]:F4},{q[i][1]:F4},{q[i][2]:F4},{q[i][3]:F4},{q[i][4]:F4}]");
        }
        for (int j = 0; j < d.Length; j++)
        {
            float n = 0;
            for (int k = 0; k < d[j].Length; k++) n += d[j][k] * d[j][k];
            Console.WriteLine($"|d[{j}]|^2 = {n:F6}  first5=[{d[j][0]:F4},{d[j][1]:F4},{d[j][2]:F4},{d[j][3]:F4},{d[j][4]:F4}]");
        }
    }
}
