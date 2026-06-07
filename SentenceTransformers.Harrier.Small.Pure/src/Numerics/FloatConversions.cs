using System.Runtime.CompilerServices;

namespace SentenceTransformers.Harrier.Small.Pure.Numerics;

/// <summary>
/// Conversions between the storage dtypes used by the safetensors weights (bfloat16, float16,
/// float32) and <see cref="float"/>. All conversions are branch-light and allocation-free so they
/// can be used on the load-time hot path that dequantizes ~270M parameters.
/// </summary>
internal static class FloatConversions
{
    /// <summary>Widens a bfloat16 (stored in the low 16 bits) to a float. bfloat16 is simply the top
    /// 16 bits of an IEEE-754 float, so widening is a left shift back into place.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float BFloat16ToSingle(ushort bf) => BitConverter.UInt32BitsToSingle((uint)bf << 16);

    /// <summary>Narrows a float to bfloat16 with round-to-nearest-even. Used to reproduce the
    /// bfloat16 rounding the reference model applies to the embedding scale factor.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort SingleToBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value))
        {
            return (ushort)((bits >> 16) | 0x0040u); // keep it a quiet NaN
        }
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        bits += roundingBias;
        return (ushort)(bits >> 16);
    }

    /// <summary>Widens an IEEE-754 float16 (stored in the low 16 bits) to a float.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Float16ToSingle(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);

    /// <summary>Round-trips a float through bfloat16 (round-to-nearest-even) and back to float.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float RoundToBFloat16(float value) => BFloat16ToSingle(SingleToBFloat16(value));
}
