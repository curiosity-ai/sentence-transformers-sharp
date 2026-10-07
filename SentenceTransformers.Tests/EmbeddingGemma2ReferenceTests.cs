using SentenceTransformers.EmbeddingGemma2;
using SentenceTransformers.EmbeddingGemma2.Audio;
using SentenceTransformers.EmbeddingGemma2.LiteRt;
using SentenceTransformers.EmbeddingGemma2.Model;
using SentenceTransformers.EmbeddingGemma2.Numerics;
using SentenceTransformers.EmbeddingGemma2.Vision;
using SentenceTransformers.Tests.Support;
using static SentenceTransformers.Tests.Support.EmbeddingGemma2TestAssets;

namespace SentenceTransformers.Tests;

/// <summary>
/// Layer-by-layer verification of the EmbeddingGemma 2 port against the TFLite graphs inside the
/// <c>.litertlm</c> bundles, executed by the LiteRT interpreter and dumped by
/// <c>scripts/generate_embeddinggemma2_reference.py</c>
/// (<c>--parts text,vision,audio</c>; see the script header for the exact command).
/// <para>
/// Every layer is <b>teacher-forced</b>: it is fed the reference input of that layer, so each comparison
/// isolates a single layer's error instead of accumulating it. Bit-exact agreement is impossible by
/// design: the reference quantizes activations to int8 on every fully connected layer, and a one-ulp
/// difference upstream flips a rounding decision. The tolerances therefore separate "same math, different
/// rounding" (≈1e-7 typical, a few 1e-3 where an int8 rounding flips) from a real bug (≥ 1e-1).
/// </para>
/// <para>
/// Opt-in: set <c>EMBEDDINGGEMMA2_REFERENCE_DIR</c> to the dump directories and make the bundles available
/// (see <see cref="EmbeddingGemma2TestAssets"/>); otherwise the tests return early.
/// </para>
/// </summary>
public class EmbeddingGemma2ReferenceTests
{
    private static readonly ParallelOptions Po = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

    private static LiteRtLmFile OpenAny(params EmbeddingGemma2Model[] models)
    {
        foreach (var m in models)
        {
            var path = ModelPath(m);
            if (path is not null)
            {
                return LiteRtLmFile.Open(path);
            }
        }
        return null;
    }

    [Fact]
    public void Text_EmbeddingsPerLayerInputsAndEveryLayer()
    {
        var r = EmbeddingGemma2Reference.Instance;
        if (r is null || !r.Has("text/case0/ids"))
        {
            return;
        }
        using var file = OpenAny(EmbeddingGemma2Model.Text270M, EmbeddingGemma2Model.TextVision440M, EmbeddingGemma2Model.Multimodal740M);
        if (file is null)
        {
            return;
        }
        var encoder = TextEncoder.Load(file);
        Assert.Equal(24, encoder.NumLayers);

        for (int c = 0; r.Has($"text/case{c}/ids"); c++)
        {
            var p = $"text/case{c}/";
            var ids = r.I(p + "ids");
            int n = ids.Length;

            // Token embedding (int4 table · scale · √512) must be exact.
            var x = new float[n * 512];
            for (int t = 0; t < n; t++)
            {
                encoder.Embedder.Lookup(ids[t], x.AsSpan(t * 512, 512));
            }
            Assert.Equal(r.F(p + "embeddings"), x);

            // Free-running pass: per-layer inputs are computed before any layer runs.
            float[] ple = null;
            var outs = encoder.Forward(new[] { ids }, Po, null, (l, t) =>
            {
                if (l == -1)
                {
                    ple = t.AsSpan(0, n * 24 * 512).ToArray();
                }
            });
            var refPle = r.F(p + "per_layer_inputs");
            Assert.True(MaxRelative(ple, refPle) < 1e-5, $"case {c} per-layer inputs: {MaxRelative(ple, refPle):E2}");

            var worst = new List<double>();
            for (int l = 0; l < encoder.NumLayers; l++)
            {
                var xin = r.F(p + (l == 0 ? "embeddings" : $"layer{l - 1:00}"));
                encoder.RunSingleLayer(l, xin, refPle, Po);
                double rel = MaxRelative(xin, r.F(p + $"layer{l:00}"));
                Assert.True(rel < 3e-2, $"case {c} layer {l}: max-rel {rel:E2}");
                worst.Add(rel);
            }
            worst.Sort();
            Assert.True(worst[worst.Count / 2] < 1e-4, $"case {c}: median layer max-rel {worst[worst.Count / 2]:E2}");

            double cos = Cosine(outs[0], r.F(p + "encodings"));
            Assert.True(cos > 0.995, $"case {c}: free-running encoding cosine {cos:F5}");
        }
    }

    private static readonly string[] VisionImages = { "synthetic_528x528.png", "synthetic_640x480.png", "synthetic_300x900.png" };

    [Fact]
    public void Vision_PreprocessingEveryLayerPoolingAndAdapter()
    {
        var r = EmbeddingGemma2Reference.Instance;
        if (r is null || !r.Has("vision/end_of_vision"))
        {
            return;
        }
        using var file = OpenAny(EmbeddingGemma2Model.TextVision440M, EmbeddingGemma2Model.Multimodal740M);
        if (file is null)
        {
            return;
        }
        var vision = VisionEncoder.Load(file);
        Assert.Equal(r.F("vision/end_of_vision"), vision.EndOfVision);

        for (int img = 0; img < VisionImages.Length; img++)
        {
            var image = EmbeddingGemma2Image.FromFile(Fixture(VisionImages[img]));
            foreach (int tokens in new[] { 140, 70 })
            {
                var p = $"vision/img{img}_t{tokens}/";
                if (!r.Has(p + "patches"))
                {
                    continue;
                }
                // Decoding (stb_image) + sRGB Catmull-Rom resize (stb_image_resize v0.97) + patchify: exact.
                var pre = ImagePreprocessor.Preprocess(image, tokens, 16, 3);
                var resized = StbImageResize.ResizeRgb(image.Rgb, image.Width, image.Height, pre.Width, pre.Height);
                Assert.Equal(r.I(p + "resized_rgb"), resized.Select(b => (int)b).ToArray());
                Assert.Equal(r.F(p + "patches"), pre.Patches);
                Assert.Equal(r.I(p + "positions"), pre.Positions);

                // Patch embedding (+ factorized 2-D position tables).
                float[] embeddings = null;
                var soft = vision.Encode(pre.Patches, pre.Positions, pre.NumPatches, Po, (l, t) =>
                {
                    if (l < 0)
                    {
                        embeddings = (float[])t.Clone();
                    }
                });
                var refEmb = r.F(p + "embeddings");
                Assert.True(MaxRelative(embeddings.AsSpan(0, refEmb.Length), refEmb) < 1e-5, $"{p} patch embeddings");
                Assert.Equal(r.I(p + "mask").Sum(), soft.Count);

                // Teacher-forced transformer layers (static int8: one flipped rounding moves a value by a
                // whole quantization step, so compare by relative RMS).
                for (int l = 0; l < vision.Layers.Length; l++)
                {
                    var xin = r.F(p + (l == 0 ? "embeddings" : $"layer{l - 1:00}"));
                    vision.RunSingleLayer(l, xin, pre.Positions, pre.NumPatches, Po);
                    double rel = RelativeRms(xin, r.F(p + $"layer{l:00}"));
                    Assert.True(rel < 5e-2, $"{p} layer {l}: rel-rms {rel:E2}");
                }

                // Pooling + soft-token adapter from the reference last layer.
                var pooled = vision.PoolForTest(r.F(p + $"layer{vision.Layers.Length - 1:00}"), pre.Positions, pre.NumPatches, Po);
                var refSoft = r.F(p + "soft_tokens");
                Assert.True(MaxRelative(pooled.Embeddings.AsSpan(0, refSoft.Length), refSoft) < 1e-5, $"{p} pool + adapter");
            }
        }
    }

    [Fact]
    public void Audio_FrontEndEveryOpAndEveryChunk()
    {
        var r = EmbeddingGemma2Reference.Instance;
        if (r is null || !r.Has("audio/end_of_audio"))
        {
            return;
        }
        using var file = OpenAny(EmbeddingGemma2Model.Multimodal740M);
        if (file is null)
        {
            return;
        }
        var audio = AudioEncoder.Load(file, file.ReadEmbeddingMetadata());
        Assert.Equal(r.F("audio/end_of_audio"), audio.EndOfAudio);
        var model = file.ReadModel(AudioEncoder.EncoderModelType);
        var graph = model.Subgraph(model.Signatures[0].SubgraphIndex);

        for (int c = 0; r.Has($"audio/clip{c}/pcm"); c++)
        {
            var p = $"audio/clip{c}/";
            var refMel = r.F(p + "mel");
            var mel = audio.Frontend.Compute(r.F(p + "pcm"), out int frames);
            Assert.Equal(refMel.Length / audio.MelBins, frames);
            Assert.True(MaxRelative(mel, refMel) < 1e-5, $"clip {c} log-mel: {MaxRelative(mel, refMel):E2}");

            // Every op of the streaming Conformer, each fed the reference value of the previous op's
            // output. Chunk k > 0 starts from the (reference) state outputs of chunk k - 1, so this also
            // verifies the streaming state hand-over.
            var exec = new GraphExecutor(model, model.Signatures[0].Key);
            Dictionary<string, GraphTensor> previous = null;
            int window = exec.TensorInfo(exec.InputIndices["segment_values"]).Shape[1];
            int stride = window - (int)exec.TensorInfo(exec.InputIndices["feature_state_0"]).ElementCount;
            for (int k = 0; r.Has(p + $"chunk{k}/op_outputs"); k++)
            {
                var opOut = r.F(p + $"chunk{k}/op_outputs");
                var sizes = r.I(p + $"chunk{k}/op_sizes");
                var offsets = new int[sizes.Length];
                for (int i = 1; i < sizes.Length; i++)
                {
                    offsets[i] = offsets[i - 1] + sizes[i - 1];
                }
                var inputs = new Dictionary<string, GraphTensor>();
                foreach (var (name, idx) in exec.InputIndices)
                {
                    var info = exec.TensorInfo(idx);
                    inputs[name] = previous is not null && previous.TryGetValue(name, out var state) ? state
                        : info.Type == TfLiteType.Bool ? GraphTensor.Bool(info.Shape) : GraphTensor.Float(info.Shape);
                }
                int len = Math.Min(window, frames - k * stride);
                inputs["segment_values"] = GraphTensor.Float(new[] { 1, window, audio.MelBins });
                inputs["segment_mask"] = GraphTensor.Bool(new[] { 1, window });
                Array.Copy(refMel, k * stride * audio.MelBins, inputs["segment_values"].F, 0, len * audio.MelBins);
                Array.Fill(inputs["segment_mask"].B, true, 0, len);
                var failures = new List<string>();
                int checkedOps = 0;
                previous = exec.Run(inputs, Po, (oi, values) =>
                {
                    var t = values[graph.Operators[oi].Outputs[0]];
                    var expected = opOut.AsSpan(offsets[oi], sizes[oi]);
                    float[] mine = t.F ?? (t.I is not null ? t.I.Select(v => (float)v).ToArray() : t.B.Select(v => v ? 1f : 0f).ToArray());
                    if (mine.Length != expected.Length)
                    {
                        failures.Add($"op {oi}: {mine.Length} values, expected {expected.Length}");
                        return;
                    }
                    double rel = MaxRelative(mine, expected);
                    if (rel > 1e-5)
                    {
                        failures.Add($"op {oi} ({graph.Operators[oi].CompositeName ?? graph.Operators[oi].Opcode.ToString()}): max-rel {rel:E2}");
                    }
                    checkedOps++;
                    if (t.F is not null)
                    {
                        values[graph.Operators[oi].Outputs[0]] = GraphTensor.Float(t.Shape, expected.ToArray());
                    }
                });
                Assert.True(failures.Count == 0, $"clip {c} chunk {k}:\n" + string.Join("\n", failures.Take(20)));
                Assert.Equal(sizes.Length, checkedOps);
            }

            // Whole clip, free-running from the reference features. Static int8 activations make the
            // 12-block stack chaotic: one flipped rounding moves a value by a full quantization step and the
            // difference is carried forward through the cached attention / convolution state, so free-running
            // tokens drift away from the interpreter's (cosine ≈ 0.96-0.99) even though every op above agrees
            // to 1e-5. The interpreter itself only agrees with the LiteRT-LM engine to ≈ 0.997 on the final
            // embedding, which is the level EmbeddingGemma2ModelTests.Audio_MatchesEngine checks; this only
            // guards against gross errors (wrong chunking, lost state, misordered tokens).
            int chunks = 0;
            double worstChunk = 0;
            var tokens = audio.EncodeMel(refMel, frames, out int count, Po, (k, outs) =>
            {
                worstChunk = Math.Max(worstChunk, RelativeRms(outs["features"].F, r.F(p + $"chunk{k}/features")));
                chunks++;
            });
            Assert.True(worstChunk < 0.6, $"clip {c}: worst chunk features rel-rms {worstChunk:E2}");
            var refTokens = r.F(p + "tokens");
            Assert.Equal(refTokens.Length / audio.OutputSize, count);
            double tokenCos = Cosine(tokens, refTokens);
            Assert.True(tokenCos > 0.9, $"clip {c} tokens: cosine {tokenCos:F5}");
            Assert.True(chunks >= 3);
        }
    }
}
