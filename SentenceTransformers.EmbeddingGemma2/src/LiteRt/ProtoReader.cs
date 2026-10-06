using System.Buffers.Binary;
using System.Text;

namespace SentenceTransformers.EmbeddingGemma2.LiteRt;

/// <summary>
/// A tiny forward-only protobuf wire-format reader, used for the SentencePiece <c>ModelProto</c> and the
/// LiteRT-LM <c>EmbeddingMetadata</c> - the only two protobuf messages this package needs to decode.
/// </summary>
internal ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _pos;

    public ProtoReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _pos = 0;
    }

    public bool End => _pos >= _data.Length;

    /// <summary>Reads the next field key. Returns false at the end of the message.</summary>
    public bool Next(out int field, out int wireType)
    {
        if (_pos >= _data.Length)
        {
            field = 0;
            wireType = 0;
            return false;
        }
        ulong key = ReadVarint();
        field = (int)(key >> 3);
        wireType = (int)(key & 7);
        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        int shift = 0;
        while (true)
        {
            byte b = _data[_pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return result;
            }
            shift += 7;
            if (shift > 63)
            {
                throw new InvalidDataException("Malformed protobuf varint.");
            }
        }
    }

    public int ReadInt32() => (int)ReadVarint();
    public bool ReadBool() => ReadVarint() != 0;

    public float ReadFloat()
    {
        float v = BinaryPrimitives.ReadSingleLittleEndian(_data.Slice(_pos, 4));
        _pos += 4;
        return v;
    }

    public ReadOnlySpan<byte> ReadBytes()
    {
        int len = checked((int)ReadVarint());
        var s = _data.Slice(_pos, len);
        _pos += len;
        return s;
    }

    public string ReadString() => Encoding.UTF8.GetString(ReadBytes());

    public void Skip(int wireType)
    {
        switch (wireType)
        {
            case 0: ReadVarint(); break;
            case 1: _pos += 8; break;
            case 2: ReadBytes(); break;
            case 5: _pos += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
        }
    }
}
