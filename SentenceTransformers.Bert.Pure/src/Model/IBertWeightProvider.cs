namespace SentenceTransformers.Bert.Pure.Model;

/// <summary>Abstracts where a BERT's frozen weights are read from (HF safetensors, or the fp32 tensors
/// extracted from an embedded ONNX graph), so <see cref="BertModel"/> loading is source-agnostic.</summary>
internal interface IBertWeightProvider
{
    bool    Contains(string name);
    float[] Read(string name);
}
