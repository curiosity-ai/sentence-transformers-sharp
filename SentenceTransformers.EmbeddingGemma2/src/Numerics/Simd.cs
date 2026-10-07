using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>Vector-width choices shared by the kernels.</summary>
internal static class Simd
{
    /// <summary>
    /// Use 512-bit vectors whenever the CPU has AVX-512 (F/BW/VL), as XNNPACK does. .NET's
    /// <c>Vector512.IsHardwareAccelerated</c> reports false on some Intel parts by default (it prefers 256 bits
    /// where wide vectors may lower clock speeds), but these kernels are all 512-bit-heavy, where the wider
    /// registers win. Element-wise results are identical at any width.
    /// </summary>
    public static readonly bool Use512 = Avx512F.IsSupported && Avx512BW.IsSupported && Avx512F.VL.IsSupported;
}
