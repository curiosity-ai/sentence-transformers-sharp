using SentenceTransformers.Harrier.Small.Pure;

namespace SentenceTransformers.Tests;

/// <summary>
/// End-to-end test for the pure-managed encoder. It downloads the ~540 MB bfloat16 safetensors
/// weights and reproduces the query/document similarity matrix published on the model card
/// (microsoft/harrier-oss-v1-270m), which is the strongest available parity check against the
/// reference implementation.
///
/// It is opt-in - set the environment variable <c>HARRIER_PURE_E2E=1</c> to run it - so the normal
/// test run stays fast and offline. It deliberately uses the public <see cref="SentenceEncoder.CreateAsync"/>
/// path: the encoder loads its embedded Gemma tokenizer, so it stays correct even though this test
/// project references other encoder packages that ship their own <c>Resources/tokenizer.json</c>.
/// </summary>
public class PureEncoderEndToEndTests
{
    [Fact]
    public async Task ReproducesModelCardScoreMatrix()
    {
        // Opt-in: skip unless explicitly enabled, so the normal suite stays fast and offline.
        // (xUnit 2.x has no dynamic Assert.Skip, so this returns early instead.)
        if (Environment.GetEnvironmentVariable("HARRIER_PURE_E2E") != "1")
        {
            return;
        }

        using var enc = await SentenceEncoder.CreateAsync();

        var queries = new[] { "how much protein should a female eat", "summit define" };
        var documents = new[]
        {
            "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        };

        var q = await enc.EncodeQueriesAsync(queries, SentenceEncoder.Prompts.WebSearchQuery);
        var d = await enc.EncodeAsync(documents);

        Assert.All(q, v => Assert.Equal(640, v.Length));

        float[,] expected = { { 64.57f, 25.45f }, { 29.97f, 67.41f } };
        for (int i = 0; i < q.Length; i++)
        {
            for (int j = 0; j < d.Length; j++)
            {
                float dot = 0;
                for (int k = 0; k < q[i].Length; k++) dot += q[i][k] * d[j][k];
                float score = dot * 100f;
                Assert.True(MathF.Abs(score - expected[i, j]) < 1.0f,
                    $"score[{i},{j}]={score:F2} expected ~{expected[i, j]:F2}");
            }
        }
    }
}
