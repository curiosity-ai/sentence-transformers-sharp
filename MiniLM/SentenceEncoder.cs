using BERTTokenizers;
using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static  MiniLM.NaiveHelpers;

namespace MiniLM;

public class SentenceEncoder
{
    private InferenceSession _session;
    private TokenizerBase _tokenizer;

    public SentenceEncoder()
    {
        var modelPath = "all-MiniLM-L6-v2/model.onnx";
        var hiddenSize = 384;
        _session = new InferenceSession(modelPath);
        _tokenizer = new BertBaseTokenizer();
    }

    public DenseTensor<float> Encode(string[] sentences)
    {
        var numSentences = sentences.Length;

        var encoded = _tokenizer.Encode(sentences);
        var tokenCount = encoded.First().InputIds.Length;

        var attentionMask = new DenseTensor<long>(encoded.SelectMany(e => e.AttentionMask).ToArray(), new[] { numSentences, tokenCount });
        var attentionMaskFloat = new DenseTensor<float>(encoded.SelectMany(e => Array.ConvertAll(e.AttentionMask, i => (float)i)).ToArray(), new[] { numSentences, tokenCount });
        var input = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor<long>("input_ids", new DenseTensor<long>(encoded.SelectMany(e => e.InputIds).ToArray(), new[] { numSentences, tokenCount })),
            NamedOnnxValue.CreateFromTensor<long>("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor<long>("token_type_ids", new DenseTensor<long>(encoded.SelectMany(e => e.TokenTypeIds).ToArray(), new[] { numSentences, tokenCount }))
        };
        var output = _session.Run(input);
        var output_pooled = MeanPooling((DenseTensor<float>)output.First().Value, attentionMaskFloat);
        var output_pooled_normalized = Normalize(output_pooled);
        return output_pooled_normalized;

    }
}