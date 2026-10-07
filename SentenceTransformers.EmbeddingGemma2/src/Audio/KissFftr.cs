namespace SentenceTransformers.EmbeddingGemma2.Audio;

/// <summary>
/// Single-precision real FFT ported from KISS FFT (<c>kiss_fft.c</c> / <c>kiss_fftr.c</c>, Mark Borgerding,
/// BSD-3-Clause), the FFT LiteRT-LM's audio front-end uses (<c>kiss_fftr</c> with <c>kiss_fft_scalar = float</c>).
/// The mixed-radix decomposition, twiddle tables (computed in double, rounded to float) and every float
/// operation are reproduced in the same order without fused multiply-adds, so the spectrum is bit-identical to
/// the engine's.
/// </summary>
internal sealed class KissFftr
{
    private readonly int _nfft;          // complex FFT size (= real length / 2)
    private readonly float[] _twR, _twI; // kiss_fft twiddles
    private readonly float[] _superR, _superI;
    private readonly int[] _factors;

    public int RealLength => 2 * _nfft;

    public KissFftr(int realLength)
    {
        if (realLength <= 0 || (realLength & 1) != 0)
        {
            throw new ArgumentException("Real FFT length must be positive and even.", nameof(realLength));
        }
        _nfft = realLength / 2;
        _twR = new float[_nfft];
        _twI = new float[_nfft];
        for (int i = 0; i < _nfft; i++)
        {
            const double Pi = 3.141592653589793238462643383279502884197169399375105820974944;
            double phase = -2 * Pi * i / _nfft;
            _twR[i] = (float)Math.Cos(phase);
            _twI[i] = (float)Math.Sin(phase);
        }
        _superR = new float[_nfft / 2];
        _superI = new float[_nfft / 2];
        for (int i = 0; i < _nfft / 2; i++)
        {
            double phase = -3.14159265358979323846264338327 * ((double)(i + 1) / _nfft + .5);
            _superR[i] = (float)Math.Cos(phase);
            _superI[i] = (float)Math.Sin(phase);
        }
        _factors = Factor(_nfft);
    }

    /// <summary><c>kf_factor</c>: radix-4 first, then 2, then odd primes.</summary>
    private static int[] Factor(int n)
    {
        var f = new List<int>();
        int p = 4;
        double floorSqrt = Math.Floor(Math.Sqrt(n));
        do
        {
            while (n % p != 0)
            {
                p = p switch { 4 => 2, 2 => 3, _ => p + 2 };
                if (p > floorSqrt)
                {
                    p = n;
                }
            }
            n /= p;
            f.Add(p);
            f.Add(n);
        }
        while (n > 1);
        return f.ToArray();
    }

    /// <summary><c>kiss_fftr</c>: <paramref name="input"/> has <see cref="RealLength"/> samples; writes
    /// <c>RealLength / 2 + 1</c> bins.</summary>
    public void Forward(ReadOnlySpan<float> input, Span<float> outR, Span<float> outI)
    {
        int n = _nfft;
        var inR = new float[n];
        var inI = new float[n];
        for (int i = 0; i < n; i++)
        {
            inR[i] = input[2 * i];
            inI[i] = input[2 * i + 1];
        }
        var fr = new float[n];
        var fi = new float[n];
        Work(fr, fi, 0, inR, inI, 0, 1, 0);

        float tdcR = fr[0], tdcI = fi[0];
        outR[0] = tdcR + tdcI;
        outR[n] = tdcR - tdcI;
        outI[0] = 0f;
        outI[n] = 0f;
        for (int k = 1; k <= n / 2; k++)
        {
            float fpkR = fr[k], fpkI = fi[k];
            float fpnkR = fr[n - k], fpnkI = -fi[n - k];
            float f1kR = fpkR + fpnkR, f1kI = fpkI + fpnkI;
            float f2kR = fpkR - fpnkR, f2kI = fpkI - fpnkI;
            float swR = _superR[k - 1], swI = _superI[k - 1];
            float twR = f2kR * swR - f2kI * swI;
            float twI = f2kR * swI + f2kI * swR;
            outR[k] = (f1kR + twR) * 0.5f;
            outI[k] = (f1kI + twI) * 0.5f;
            outR[n - k] = (f1kR - twR) * 0.5f;
            outI[n - k] = (twI - f1kI) * 0.5f;
        }
    }

    /// <summary><c>kf_work</c>: recursive decimation in time, then the stage butterfly.</summary>
    private void Work(float[] or, float[] oi, int fout, float[] inR, float[] inI, int f, int fstride, int fi)
    {
        int p = _factors[fi], m = _factors[fi + 1];
        if (m == 1)
        {
            for (int j = 0; j < p; j++)
            {
                or[fout + j] = inR[f];
                oi[fout + j] = inI[f];
                f += fstride;
            }
        }
        else
        {
            for (int j = 0; j < p; j++)
            {
                Work(or, oi, fout + j * m, inR, inI, f, fstride * p, fi + 2);
                f += fstride;
            }
        }
        switch (p)
        {
            case 2: Bfly2(or, oi, fout, fstride, m); break;
            case 3: Bfly3(or, oi, fout, fstride, m); break;
            case 4: Bfly4(or, oi, fout, fstride, m); break;
            case 5: Bfly5(or, oi, fout, fstride, m); break;
            default: BflyGeneric(or, oi, fout, fstride, m, p); break;
        }
    }

    // C_MUL(m, a, b): (a.r·b.r − a.i·b.i, a.r·b.i + a.i·b.r)
    private static void Mul(float ar, float ai, float br, float bi, out float r, out float i)
    {
        r = ar * br - ai * bi;
        i = ar * bi + ai * br;
    }

    private void Bfly2(float[] r, float[] im, int fout, int fstride, int m)
    {
        for (int k = 0, tw = 0; k < m; k++, tw += fstride)
        {
            int a = fout + k, b = a + m;
            Mul(r[b], im[b], _twR[tw], _twI[tw], out float tr, out float ti);
            r[b] = r[a] - tr;
            im[b] = im[a] - ti;
            r[a] += tr;
            im[a] += ti;
        }
    }

    private void Bfly4(float[] r, float[] im, int fout, int fstride, int m)
    {
        for (int k = 0; k < m; k++)
        {
            int a0 = fout + k, a1 = a0 + m, a2 = a0 + 2 * m, a3 = a0 + 3 * m;
            int t1 = k * fstride, t2 = 2 * k * fstride, t3 = 3 * k * fstride;
            Mul(r[a1], im[a1], _twR[t1], _twI[t1], out float s0r, out float s0i);
            Mul(r[a2], im[a2], _twR[t2], _twI[t2], out float s1r, out float s1i);
            Mul(r[a3], im[a3], _twR[t3], _twI[t3], out float s2r, out float s2i);
            float s5r = r[a0] - s1r, s5i = im[a0] - s1i;
            r[a0] += s1r;
            im[a0] += s1i;
            float s3r = s0r + s2r, s3i = s0i + s2i;
            float s4r = s0r - s2r, s4i = s0i - s2i;
            r[a2] = r[a0] - s3r;
            im[a2] = im[a0] - s3i;
            r[a0] += s3r;
            im[a0] += s3i;
            r[a1] = s5r + s4i;
            im[a1] = s5i - s4r;
            r[a3] = s5r - s4i;
            im[a3] = s5i + s4r;
        }
    }

    private void Bfly3(float[] r, float[] im, int fout, int fstride, int m)
    {
        float epi3i = _twI[fstride * m];
        for (int k = 0; k < m; k++)
        {
            int a0 = fout + k, a1 = a0 + m, a2 = a0 + 2 * m;
            int t1 = k * fstride, t2 = 2 * k * fstride;
            Mul(r[a1], im[a1], _twR[t1], _twI[t1], out float s1r, out float s1i);
            Mul(r[a2], im[a2], _twR[t2], _twI[t2], out float s2r, out float s2i);
            float s3r = s1r + s2r, s3i = s1i + s2i;
            float s0r = s1r - s2r, s0i = s1i - s2i;
            r[a1] = r[a0] - s3r * 0.5f;
            im[a1] = im[a0] - s3i * 0.5f;
            s0r *= epi3i;
            s0i *= epi3i;
            r[a0] += s3r;
            im[a0] += s3i;
            r[a2] = r[a1] + s0i;
            im[a2] = im[a1] - s0r;
            r[a1] -= s0i;
            im[a1] += s0r;
        }
    }

    private void Bfly5(float[] r, float[] im, int fout, int fstride, int m)
    {
        float yar = _twR[fstride * m], yai = _twI[fstride * m];
        float ybr = _twR[fstride * 2 * m], ybi = _twI[fstride * 2 * m];
        for (int u = 0; u < m; u++)
        {
            int a0 = fout + u, a1 = a0 + m, a2 = a0 + 2 * m, a3 = a0 + 3 * m, a4 = a0 + 4 * m;
            float s0r = r[a0], s0i = im[a0];
            Mul(r[a1], im[a1], _twR[u * fstride], _twI[u * fstride], out float s1r, out float s1i);
            Mul(r[a2], im[a2], _twR[2 * u * fstride], _twI[2 * u * fstride], out float s2r, out float s2i);
            Mul(r[a3], im[a3], _twR[3 * u * fstride], _twI[3 * u * fstride], out float s3r, out float s3i);
            Mul(r[a4], im[a4], _twR[4 * u * fstride], _twI[4 * u * fstride], out float s4r, out float s4i);
            float s7r = s1r + s4r, s7i = s1i + s4i;
            float s10r = s1r - s4r, s10i = s1i - s4i;
            float s8r = s2r + s3r, s8i = s2i + s3i;
            float s9r = s2r - s3r, s9i = s2i - s3i;
            r[a0] += s7r + s8r;
            im[a0] += s7i + s8i;
            float s5r = s0r + s7r * yar + s8r * ybr;
            float s5i = s0i + s7i * yar + s8i * ybr;
            float s6r = s10i * yai + s9i * ybi;
            float s6i = -(s10r * yai) - s9r * ybi;
            r[a1] = s5r - s6r;
            im[a1] = s5i - s6i;
            r[a4] = s5r + s6r;
            im[a4] = s5i + s6i;
            float s11r = s0r + s7r * ybr + s8r * yar;
            float s11i = s0i + s7i * ybr + s8i * yar;
            float s12r = -(s10i * ybi) + s9i * yai;
            float s12i = s10r * ybi - s9r * yai;
            r[a2] = s11r + s12r;
            im[a2] = s11i + s12i;
            r[a3] = s11r - s12r;
            im[a3] = s11i - s12i;
        }
    }

    private void BflyGeneric(float[] r, float[] im, int fout, int fstride, int m, int p)
    {
        var scR = new float[p];
        var scI = new float[p];
        for (int u = 0; u < m; u++)
        {
            for (int q1 = 0, k = u; q1 < p; q1++, k += m)
            {
                scR[q1] = r[fout + k];
                scI[q1] = im[fout + k];
            }
            for (int q1 = 0, k = u; q1 < p; q1++, k += m)
            {
                int twidx = 0;
                float accR = scR[0], accI = scI[0];
                for (int q = 1; q < p; q++)
                {
                    twidx += fstride * k;
                    if (twidx >= _nfft)
                    {
                        twidx -= _nfft;
                    }
                    Mul(scR[q], scI[q], _twR[twidx], _twI[twidx], out float tr, out float ti);
                    accR += tr;
                    accI += ti;
                }
                r[fout + k] = accR;
                im[fout + k] = accI;
            }
        }
    }
}
