using SentenceTransformers.Stq;

namespace SentenceTransformers.Quantize;

/// <summary>Prints an <c>.stq</c> file's header: metadata, rotations, and every tensor's band, shape
/// and footprint. Reads only the header and offsets - no tensor is decoded.</summary>
public static class Inspector
{
    public static int Run(string path, TextWriter log)
    {
        var file = StqFile.Load(path);
        var info = new FileInfo(path);

        log.WriteLine($"{path}  ({info.Length / 1024.0 / 1024.0:F1} MB on disk, {file.PayloadBytes / 1024.0 / 1024.0:F1} MB of tensor data)");
        log.WriteLine();
        log.WriteLine("Metadata");
        foreach (var (k, v) in file.Metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            log.WriteLine($"  {k,-20} {v}");
        }

        if (file.Rotations.Count > 0)
        {
            log.WriteLine();
            log.WriteLine("Rotations");
            foreach (var (id, r) in file.Rotations.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                log.WriteLine($"  {id,-8} dim {r.Dim,6}  Hadamard block {r.Block,6}  blocks/vector {r.Dim / r.Block}");
            }
        }

        log.WriteLine();
        log.WriteLine($"Tensors ({file.Tensors.Count})");
        log.WriteLine($"  {"name",-48} {"band",-6} {"shape",-18} {"rotation",-9} {"MB",7} {"bits/w",6}");
        long totalBytes = 0, totalParams = 0;
        foreach (var t in file.Tensors.Values.OrderBy(t => t.Name, StringComparer.Ordinal))
        {
            long bytes = (t.CodesEnd - t.CodesBegin) + (t.ScalesEnd - t.ScalesBegin) + (t.DataEnd - t.DataBegin);
            totalBytes += bytes;
            totalParams += t.ElementCount;
            log.WriteLine($"  {t.Name,-48} {StqFormat.BandName(t.Band),-6} {"[" + string.Join(", ", t.Shape) + "]",-18} " +
                          $"{t.RotationId ?? "-",-9} {bytes / 1024.0 / 1024.0,7:F2} {(double)bytes * 8 / t.ElementCount,6:F3}");
        }

        log.WriteLine();
        log.WriteLine($"  {totalParams:N0} parameters in {totalBytes / 1024.0 / 1024.0:F1} MB = {(double)totalBytes * 8 / totalParams:F3} bits/weight");
        return 0;
    }
}
