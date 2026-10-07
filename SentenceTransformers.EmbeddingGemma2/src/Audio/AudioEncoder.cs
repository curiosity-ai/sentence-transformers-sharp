using System.Numerics.Tensors;
using SentenceTransformers.EmbeddingGemma2.LiteRt;

namespace SentenceTransformers.EmbeddingGemma2.Audio;

/// <summary>
/// The EmbeddingGemma 2 audio tower (<c>tf_lite_audio_encoder_hw</c> + <c>tf_lite_audio_adapter</c> +
/// <c>tf_lite_end_of_audio</c>): a streaming Conformer (2× strided conv subsampling, 12 blocks of
/// macaron FFN, chunked local self-attention with relative positions and logit soft-capping, GLU
/// depthwise-conv module) over 16 kHz log-mel features, emitting 25 tokens per second of audio.
/// <para>
/// The encoder graph is executed by <see cref="GraphExecutor"/> chunk by chunk exactly like LiteRT-LM's
/// <c>AudioLiteRtCompiledModelExecutor::EncodeSpecsAndMasks</c>: windows of <c>segment</c> mel frames
/// that overlap by <c>overlap</c> frames (51 / 3), the last partial window zero-padded and masked, and
/// every state output (cached keys/queries/values, conv padding, lookback features, masks) fed back as
/// the next chunk's input. State starts zeroed for every clip.
/// </para>
/// </summary>
internal sealed class AudioEncoder
{
    public const string EncoderModelType = "tf_lite_audio_encoder_hw";
    public const string AdapterModelType = "tf_lite_audio_adapter";
    public const string EndOfAudioModelType = "tf_lite_end_of_audio";

    private const string SegmentValues = "segment_values";
    private const string SegmentMask = "segment_mask";

    private readonly GraphExecutor _encoder;
    private readonly GraphExecutor _adapter;
    private readonly string[] _stateNames;
    private readonly int _window;
    private readonly int _overlap;
    private readonly int _tokensPerChunk;

    public LogMelFrontend Frontend { get; }
    public float[] EndOfAudio { get; }
    public int OutputSize { get; }
    public int MelBins { get; }

    private AudioEncoder(GraphExecutor encoder, GraphExecutor adapter, LogMelFrontend frontend, float[] eoa)
    {
        _encoder = encoder;
        _adapter = adapter;
        Frontend = frontend;
        EndOfAudio = eoa;
        var seg = encoder.TensorInfo(encoder.InputIndices[SegmentValues]);
        _window = seg.Shape[1];
        MelBins = seg.Shape[2];
        // The overlap is the size of the subsampling feature state (3 for this model), like the engine.
        _overlap = (int)encoder.TensorInfo(encoder.InputIndices["feature_state_0"]).ElementCount;
        _tokensPerChunk = encoder.TensorInfo(encoder.OutputIndices["features"]).Shape[1];
        _stateNames = encoder.InputIndices.Keys.Where(n => n != SegmentValues && n != SegmentMask && encoder.OutputIndices.ContainsKey(n)).ToArray();
        OutputSize = adapter.TensorInfo(adapter.OutputIndices.Values.First()).Shape[^1];
    }

    public static AudioEncoder Load(LiteRtLmFile file, EmbeddingMetadata metadata)
    {
        var encoderModel = file.ReadModel(EncoderModelType);
        var encoder = new GraphExecutor(encoderModel, encoderModel.Signatures[0].Key);
        var adapterModel = file.ReadModel(AdapterModelType);
        var adapter = new GraphExecutor(adapterModel, adapterModel.Signatures[0].Key);
        var eoaModel = file.ReadModel(EndOfAudioModelType);
        var esg = eoaModel.Subgraph(eoaModel.Signatures[0].SubgraphIndex);
        var eoa = eoaModel.ReadFloats(esg.Tensors[eoaModel.Signatures[0].Outputs.Values.First()]);
        var config = metadata.Audio ?? throw new InvalidDataException("The bundle has no audio preprocessor configuration.");
        return new AudioEncoder(encoder, adapter, new LogMelFrontend(config), eoa);
    }

    /// <summary>Encodes 16 kHz mono PCM to audio soft tokens [count, 512] in the text embedding space.</summary>
    public float[] Encode(ReadOnlySpan<float> pcm16k, out int count, ParallelOptions po)
    {
        var mel = Frontend.Compute(pcm16k, out int frames);
        return EncodeMel(mel, frames, out count, po);
    }

    /// <summary>Runs the streaming encoder + adapter over [frames, mel] features (test entry point).</summary>
    public float[] EncodeMel(float[] mel, int frames, out int count, ParallelOptions po, Action<int, Dictionary<string, GraphTensor>> chunkHook = null)
    {
        int stride = _window - _overlap;
        var state = new Dictionary<string, GraphTensor>(StringComparer.Ordinal);
        foreach (var name in _stateNames)
        {
            var info = _encoder.TensorInfo(_encoder.InputIndices[name]);
            state[name] = info.Type == TfLiteType.Bool ? GraphTensor.Bool(info.Shape) : GraphTensor.Float(info.Shape);
        }
        var tokens = new List<float>();
        count = 0;
        int chunk = 0;
        for (int pos = 0; pos + _window <= frames || pos + _overlap < frames; pos += stride, chunk++)
        {
            po.CancellationToken.ThrowIfCancellationRequested();
            int len = Math.Min(_window, frames - pos);
            var values = GraphTensor.Float(new[] { 1, _window, MelBins });
            Array.Copy(mel, pos * MelBins, values.F, 0, len * MelBins);
            var mask = GraphTensor.Bool(new[] { 1, _window });
            Array.Fill(mask.B, true, 0, len);

            var inputs = new Dictionary<string, GraphTensor>(state, StringComparer.Ordinal)
            {
                [SegmentValues] = values,
                [SegmentMask] = mask,
            };
            var outputs = _encoder.Run(inputs, po);
            var outMask = outputs["mask"];
            int valid = 0;
            for (int i = outMask.B.Length - 1; i >= 0; i--)
            {
                if (outMask.B[i])
                {
                    valid = i + 1;
                    break;
                }
            }
            var adapted = _adapter.Run(new Dictionary<string, GraphTensor> { ["features"] = outputs["features"], ["mask"] = outMask }, po);
            var proj = adapted.Values.First();
            tokens.AddRange(proj.F.AsSpan(0, valid * OutputSize).ToArray());
            count += valid;
            chunkHook?.Invoke(chunk, outputs);
            foreach (var name in _stateNames)
            {
                state[name] = outputs[name].Clone();
            }
        }
        return tokens.ToArray();
    }
}
