using System.Buffers.Binary;
using System.Text;

namespace SentenceTransformers.EmbeddingGemma2.LiteRt;

/// <summary>
/// A minimal, allocation-light reader for FlatBuffers tables - just enough to walk the LiteRT-LM header
/// and the TFLite model schema without a generated-code dependency. Field ids are the vtable slot
/// offsets from the generated schema (4 = first field, 6 = second, ...).
/// </summary>
internal readonly struct FbTable
{
    private readonly byte[] _buf;
    private readonly int _pos;

    public FbTable(byte[] buffer, int position)
    {
        _buf = buffer;
        _pos = position;
    }

    public bool IsNull => _buf is null;
    public byte[] Buffer => _buf;
    public int Position => _pos;

    /// <summary>Root table of a buffer (the first 4 bytes hold the offset to it).</summary>
    public static FbTable Root(byte[] buffer, int start = 0)
        => new(buffer, start + (int)BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(start)));

    /// <summary>Absolute position of a field, or 0 when absent from the vtable.</summary>
    private int FieldPos(int vtableOffset)
    {
        int vtable = _pos - BinaryPrimitives.ReadInt32LittleEndian(_buf.AsSpan(_pos));
        int vtableSize = BinaryPrimitives.ReadUInt16LittleEndian(_buf.AsSpan(vtable));
        if (vtableOffset >= vtableSize)
        {
            return 0;
        }
        int rel = BinaryPrimitives.ReadUInt16LittleEndian(_buf.AsSpan(vtable + vtableOffset));
        return rel == 0 ? 0 : _pos + rel;
    }

    public bool Has(int field) => FieldPos(field) != 0;

    public byte GetByte(int field, byte def = 0) { int p = FieldPos(field); return p == 0 ? def : _buf[p]; }
    public sbyte GetSByte(int field, sbyte def = 0) { int p = FieldPos(field); return p == 0 ? def : (sbyte)_buf[p]; }
    public bool GetBool(int field, bool def = false) { int p = FieldPos(field); return p == 0 ? def : _buf[p] != 0; }
    public ushort GetUInt16(int field, ushort def = 0) { int p = FieldPos(field); return p == 0 ? def : BinaryPrimitives.ReadUInt16LittleEndian(_buf.AsSpan(p)); }
    public int GetInt32(int field, int def = 0) { int p = FieldPos(field); return p == 0 ? def : BinaryPrimitives.ReadInt32LittleEndian(_buf.AsSpan(p)); }
    public uint GetUInt32(int field, uint def = 0) { int p = FieldPos(field); return p == 0 ? def : BinaryPrimitives.ReadUInt32LittleEndian(_buf.AsSpan(p)); }
    public long GetInt64(int field, long def = 0) { int p = FieldPos(field); return p == 0 ? def : BinaryPrimitives.ReadInt64LittleEndian(_buf.AsSpan(p)); }
    public ulong GetUInt64(int field, ulong def = 0) { int p = FieldPos(field); return p == 0 ? def : BinaryPrimitives.ReadUInt64LittleEndian(_buf.AsSpan(p)); }
    public float GetFloat(int field, float def = 0) { int p = FieldPos(field); return p == 0 ? def : BinaryPrimitives.ReadSingleLittleEndian(_buf.AsSpan(p)); }

    private int Deref(int p) => p + (int)BinaryPrimitives.ReadUInt32LittleEndian(_buf.AsSpan(p));

    public FbTable GetTable(int field)
    {
        int p = FieldPos(field);
        return p == 0 ? default : new FbTable(_buf, Deref(p));
    }

    public string GetString(int field)
    {
        int p = FieldPos(field);
        if (p == 0)
        {
            return null;
        }
        int s = Deref(p);
        int len = (int)BinaryPrimitives.ReadUInt32LittleEndian(_buf.AsSpan(s));
        return Encoding.UTF8.GetString(_buf, s + 4, len);
    }

    /// <summary>Returns (start, length) of a vector field; start points at the first element.</summary>
    public (int Start, int Length) GetVector(int field)
    {
        int p = FieldPos(field);
        if (p == 0)
        {
            return (0, 0);
        }
        int v = Deref(p);
        return (v + 4, (int)BinaryPrimitives.ReadUInt32LittleEndian(_buf.AsSpan(v)));
    }

    public int VectorLength(int field) => GetVector(field).Length;

    public ReadOnlySpan<byte> GetBytes(int field)
    {
        var (s, n) = GetVector(field);
        return n == 0 ? ReadOnlySpan<byte>.Empty : _buf.AsSpan(s, n);
    }

    /// <summary>Same as <see cref="GetBytes"/> but returns the absolute offset into <see cref="Buffer"/>.</summary>
    public (int Offset, int Length) GetBytesRange(int field) => GetVector(field);

    public int[] GetInt32Vector(int field)
    {
        var (s, n) = GetVector(field);
        var r = new int[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = BinaryPrimitives.ReadInt32LittleEndian(_buf.AsSpan(s + 4 * i));
        }
        return r;
    }

    public long[] GetInt64Vector(int field)
    {
        var (s, n) = GetVector(field);
        var r = new long[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = BinaryPrimitives.ReadInt64LittleEndian(_buf.AsSpan(s + 8 * i));
        }
        return r;
    }

    public float[] GetFloatVector(int field)
    {
        var (s, n) = GetVector(field);
        var r = new float[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = BinaryPrimitives.ReadSingleLittleEndian(_buf.AsSpan(s + 4 * i));
        }
        return r;
    }

    /// <summary>Element <paramref name="index"/> of a vector of tables.</summary>
    public FbTable GetTableElement(int field, int index)
    {
        var (s, n) = GetVector(field);
        if ((uint)index >= (uint)n)
        {
            throw new IndexOutOfRangeException();
        }
        return new FbTable(_buf, Deref(s + 4 * index));
    }
}
