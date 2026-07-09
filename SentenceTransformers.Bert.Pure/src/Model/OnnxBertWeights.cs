using System.Buffers.Binary;

namespace SentenceTransformers.Bert.Pure.Model;

/// <summary>
/// Extracts the full-precision (fp32) BERT weights directly from an embedded ONNX graph, with no ONNX
/// Runtime dependency — just a minimal managed protobuf reader. This lets the pure trainer/encoder reuse
/// the exact weights that already ship inside the MiniLM / ArcticXs packages, so no separate download is
/// needed and the reconstruction is lossless (the shipped graphs are fp32, not quantized).
///
/// <para>In the exported graph the embeddings, biases and LayerNorm scales keep their semantic HuggingFace
/// names, but each Linear's weight matrix was renamed to a numeric initializer and appears as the operand
/// of a <c>MatMul</c> whose result is added to the corresponding <c>*.bias</c>. We therefore recover each
/// weight by tracing <c>bias → Add → MatMul → weight initializer</c>, and transpose it from the ONNX
/// <c>[in,out]</c> layout to the PyTorch/HF <c>[out,in]</c> layout this library uses.</para>
/// </summary>
internal sealed class OnnxBertWeights : IBertWeightProvider
{
    private readonly Dictionary<string, float[]> _weights = new(StringComparer.Ordinal);

    public bool    Contains(string name) => _weights.ContainsKey(name);
    public float[] Read(string name)     => _weights.TryGetValue(name, out var w) ? w : throw new KeyNotFoundException($"ONNX weight '{name}' not found.");

    public OnnxBertWeights(byte[] onnx, BertConfig cfg)
    {
        var graph = ParseGraph(onnx);

        // Direct, semantically-named tensors.
        foreach (var name in new[]
        {
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias",
        })
        {
            _weights[name] = graph.Initializers[name].Data;
        }

        // producer map: tensor output name -> node index
        var producer = new Dictionary<string, Node>(StringComparer.Ordinal);
        foreach (var n in graph.Nodes)
            foreach (var o in n.Outputs) producer[o] = n;

        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"encoder.layer.{i}.";
            // LayerNorms + biases keep their names.
            foreach (var s in new[]
            {
                "attention.output.LayerNorm.weight", "attention.output.LayerNorm.bias",
                "output.LayerNorm.weight",           "output.LayerNorm.bias",
                "attention.self.query.bias", "attention.self.key.bias", "attention.self.value.bias",
                "attention.output.dense.bias", "intermediate.dense.bias", "output.dense.bias",
            })
            {
                _weights[p + s] = graph.Initializers[p + s].Data;
            }

            // Weight matrices, recovered by tracing each bias back through its Add + MatMul.
            TraceWeight(graph, producer, p + "attention.self.query.bias",   p + "attention.self.query.weight");
            TraceWeight(graph, producer, p + "attention.self.key.bias",     p + "attention.self.key.weight");
            TraceWeight(graph, producer, p + "attention.self.value.bias",   p + "attention.self.value.weight");
            TraceWeight(graph, producer, p + "attention.output.dense.bias", p + "attention.output.dense.weight");
            TraceWeight(graph, producer, p + "intermediate.dense.bias",     p + "intermediate.dense.weight");
            TraceWeight(graph, producer, p + "output.dense.bias",           p + "output.dense.weight");
        }
    }

    private void TraceWeight(GraphData g, Dictionary<string, Node> producer, string biasName, string hfWeightName)
    {
        // Find the Add that consumes this bias, take its other input, and walk to the producing MatMul.
        foreach (var add in g.Nodes)
        {
            if (add.OpType != "Add" || Array.IndexOf(add.Inputs, biasName) < 0) continue;
            string other = add.Inputs[0] == biasName ? add.Inputs[1] : add.Inputs[0];
            if (!producer.TryGetValue(other, out var mm)) continue;
            if (mm.OpType != "MatMul" && mm.OpType != "Gemm") continue;

            foreach (var inp in mm.Inputs)
            {
                if (!g.Initializers.TryGetValue(inp, out var t)) continue;
                if (t.Dims.Length != 2) continue;
                // ONNX MatMul weight is [in, out]; transpose to [out, in].
                int inDim = t.Dims[0], outDim = t.Dims[1];
                var w = new float[outDim * inDim];
                for (int r = 0; r < inDim; r++)
                    for (int c = 0; c < outDim; c++)
                        w[c * inDim + r] = t.Data[r * outDim + c];
                _weights[hfWeightName] = w;
                return;
            }
        }
        throw new InvalidDataException($"Could not trace weight for '{biasName}' in ONNX graph.");
    }

    // ----- minimal protobuf reader ----------------------------------------------------------------

    private sealed class TensorData { public string Name; public int[] Dims; public float[] Data; }
    private sealed class Node { public string OpType; public string[] Inputs; public string[] Outputs; }
    private sealed class GraphData { public Dictionary<string, TensorData> Initializers; public List<Node> Nodes; }

    private static GraphData ParseGraph(byte[] onnx)
    {
        // ModelProto.graph = field 7 (message)
        var span = new ReadOnlySpan<byte>(onnx);
        ReadOnlySpan<byte> graphBytes = default;
        var r = new Reader(span);
        while (!r.End)
        {
            var (field, wire) = r.Tag();
            if (field == 7 && wire == 2) { graphBytes = r.Bytes(); }
            else r.Skip(wire);
        }
        if (graphBytes.IsEmpty) throw new InvalidDataException("ONNX ModelProto has no graph.");

        var inits = new Dictionary<string, TensorData>(StringComparer.Ordinal);
        var nodes = new List<Node>();
        var gr = new Reader(graphBytes);
        while (!gr.End)
        {
            var (field, wire) = gr.Tag();
            if (field == 1 && wire == 2)      nodes.Add(ParseNode(gr.Bytes()));      // GraphProto.node
            else if (field == 5 && wire == 2) { var t = ParseTensor(gr.Bytes()); if (t.Data != null) inits[t.Name] = t; } // initializer
            else gr.Skip(wire);
        }
        return new GraphData { Initializers = inits, Nodes = nodes };
    }

    private static Node ParseNode(ReadOnlySpan<byte> bytes)
    {
        var inputs = new List<string>();
        var outputs = new List<string>();
        string opType = null;
        var r = new Reader(bytes);
        while (!r.End)
        {
            var (field, wire) = r.Tag();
            switch (field)
            {
                case 1 when wire == 2: inputs.Add(r.String());  break; // input
                case 2 when wire == 2: outputs.Add(r.String()); break; // output
                case 4 when wire == 2: opType = r.String();     break; // op_type
                default: r.Skip(wire); break;
            }
        }
        return new Node { OpType = opType, Inputs = inputs.ToArray(), Outputs = outputs.ToArray() };
    }

    private static TensorData ParseTensor(ReadOnlySpan<byte> bytes)
    {
        var dims = new List<int>();
        int dataType = 0;
        string name = null;
        ReadOnlySpan<byte> raw = default;
        float[] floatData = null;
        var r = new Reader(bytes);
        while (!r.End)
        {
            var (field, wire) = r.Tag();
            switch (field)
            {
                case 1 when wire == 0: dims.Add((int)r.Varint()); break;           // dims (unpacked)
                case 1 when wire == 2:                                             // dims (packed)
                {
                    var d = new Reader(r.Bytes());
                    while (!d.End) dims.Add((int)d.Varint());
                    break;
                }
                case 2 when wire == 0: dataType = (int)r.Varint(); break;          // data_type
                case 4 when wire == 2:                                             // float_data (packed)
                {
                    var fb = r.Bytes();
                    floatData = new float[fb.Length / 4];
                    for (int i = 0; i < floatData.Length; i++) floatData[i] = BinaryPrimitives.ReadSingleLittleEndian(fb.Slice(i * 4, 4));
                    break;
                }
                case 8 when wire == 2: name = r.String(); break;                  // name
                case 9 when wire == 2: raw = r.Bytes();  break;                   // raw_data
                default: r.Skip(wire); break;
            }
        }

        // Only fp32 tensors are relevant (data_type 1 == FLOAT); skip int64 buffers like position_ids.
        float[] data = null;
        if (dataType == 1)
        {
            if (!raw.IsEmpty)
            {
                data = new float[raw.Length / 4];
                for (int i = 0; i < data.Length; i++) data[i] = BinaryPrimitives.ReadSingleLittleEndian(raw.Slice(i * 4, 4));
            }
            else if (floatData != null) data = floatData;
        }
        return new TensorData { Name = name, Dims = dims.ToArray(), Data = data };
    }

    /// <summary>A forward-only protobuf field cursor over a message body.</summary>
    private ref struct Reader
    {
        private ReadOnlySpan<byte> _s;
        private int _pos;
        public Reader(ReadOnlySpan<byte> s) { _s = s; _pos = 0; }
        public bool End => _pos >= _s.Length;

        public ulong Varint()
        {
            ulong result = 0; int shift = 0;
            while (true)
            {
                byte b = _s[_pos++];
                result |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            return result;
        }

        public (int field, int wire) Tag()
        {
            ulong tag = Varint();
            return ((int)(tag >> 3), (int)(tag & 0x7));
        }

        public ReadOnlySpan<byte> Bytes()
        {
            int len = (int)Varint();
            var slice = _s.Slice(_pos, len);
            _pos += len;
            return slice;
        }

        public string String() => System.Text.Encoding.UTF8.GetString(Bytes());

        public void Skip(int wire)
        {
            switch (wire)
            {
                case 0: Varint(); break;
                case 1: _pos += 8; break;
                case 2: { int len = (int)Varint(); _pos += len; break; }
                case 5: _pos += 4; break;
                default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
            }
        }
    }
}
