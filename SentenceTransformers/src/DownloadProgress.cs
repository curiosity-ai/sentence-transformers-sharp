namespace SentenceTransformers;

/// <summary>
/// Progress snapshot for an in-progress model download. Passed to the
/// <c>reportProgress</c> callback on <c>SentenceEncoder.DownloadModelAsync</c> and
/// <c>CreateAsync</c> a few times per second.
/// </summary>
/// <param name="DownloadedBytes">Total bytes received so far. Resumed downloads include the bytes already on disk.</param>
/// <param name="TotalBytes">Total size in bytes from the response's <c>Content-Length</c> header, or <c>null</c> when the server didn't send one.</param>
/// <param name="Fraction">Progress in <c>[0, 1]</c> when <see cref="TotalBytes"/> is known; <c>0</c> otherwise.</param>
/// <param name="FileName">The on-disk filename being written (the final segment of <c>localPath</c>). Useful when a single logical model download spans multiple files (e.g. an ONNX graph plus its <c>.onnx_data</c>).</param>
public sealed record DownloadProgress(long DownloadedBytes, long? TotalBytes, float Fraction, string FileName);
