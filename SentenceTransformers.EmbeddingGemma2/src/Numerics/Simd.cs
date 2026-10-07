using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>Vector-width choices and buffer alignment helpers shared by the kernels.</summary>
internal static class Simd
{
    /// <summary>
    /// Use 512-bit vectors whenever the CPU has AVX-512 (F/BW/VL), as XNNPACK does. .NET's
    /// <c>Vector512.IsHardwareAccelerated</c> reports false on some Intel parts by default (it prefers 256 bits
    /// where wide vectors may lower clock speeds), but these kernels are all 512-bit-heavy, where the wider
    /// registers win. Element-wise results are identical at any width.
    /// </summary>
    public static readonly bool Use512 = Avx512F.IsSupported && Avx512BW.IsSupported && Avx512F.VL.IsSupported;

    /// <summary>Elements to skip from the start of <paramref name="array"/> to reach a 64-byte boundary at the array's
    /// current address. Stable for arrays on the pinned object heap; for any other array the GC may move it later,
    /// so the result is only a performance hint (the kernels never require alignment).</summary>
    public static int AlignOffset<T>(T[] array) where T : unmanaged
        => (int)(-(long)Marshal.UnsafeAddrOfPinnedArrayElement(array, 0) & 63) / Unsafe.SizeOf<T>();
}

/// <summary>Per-thread 64-byte aligned scratch buffers (one per element type) for packed GEMM operands.</summary>
internal static class AlignedScratch<T> where T : unmanaged
{
    [ThreadStatic]
    private static T[] _array;
    [ThreadStatic]
    private static int _offset;

    /// <summary>This thread's scratch of <paramref name="length"/> elements (contents undefined). The array lives on
    /// the pinned object heap, so its address never changes and the aligning offset is computed once. A caller must
    /// be done with the span before anything on the same thread asks for another one of the same type.</summary>
    public static Span<T> Get(int length)
    {
        int slack = 64 / Unsafe.SizeOf<T>();
        var a = _array;
        if (a is null || a.Length < length + slack)
        {
            _array = a = GC.AllocateUninitializedArray<T>(length + slack, pinned: true);
            _offset = Simd.AlignOffset(a);
        }
        return a.AsSpan(_offset, length);
    }
}
