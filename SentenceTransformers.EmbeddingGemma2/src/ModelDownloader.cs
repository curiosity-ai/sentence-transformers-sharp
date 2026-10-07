using System.Net.Http.Headers;

namespace SentenceTransformers.EmbeddingGemma2;

/// <summary>
/// Resumable, single-flight model downloader. Files are streamed to a <c>*.download</c> sibling and only
/// moved to their final name once complete, so an interrupted download can never leave a truncated file
/// that a later run would mistake for a valid model.
/// </summary>
public static class ModelDownloader
{
    private static readonly SemaphoreSlim _oneDownloadAtATime = new(1, 1);
    private static readonly HttpClient _client = new() { Timeout = TimeSpan.FromDays(1) };

    /// <summary>Returns true while a download is in progress.</summary>
    public static bool IsDownloading() => _oneDownloadAtATime.CurrentCount < 1;

    /// <summary>
    /// Downloads <paramref name="url"/> to <paramref name="localPath"/> unless the file already exists
    /// (and, when <paramref name="expectedSize"/> is given, has exactly that size). Retries with HTTP range
    /// requests when the connection drops.
    /// </summary>
    public static async Task DownloadFileAsync(string url, string localPath, Action<DownloadProgress> reportProgress = null, long? expectedSize = null, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(localPath))
        {
            throw new ArgumentException("Local path must be non-empty.", nameof(localPath));
        }
        if (File.Exists(localPath))
        {
            if (expectedSize is not long size || new FileInfo(localPath).Length == size)
            {
                return;
            }
            // A cached file with the wrong size is a leftover of an older/corrupt download.
            File.Delete(localPath);
        }
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme is not ("https" or "http"))
        {
            throw new InvalidOperationException($"Invalid model URL: '{url}'. Use a valid http(s) URL.");
        }
        var directory = Path.GetDirectoryName(Path.GetFullPath(localPath));
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var fileName = Path.GetFileName(localPath);
        var tempPath = localPath + ".download";
        long? totalBytes = null;

        await _oneDownloadAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (File.Exists(localPath))
            {
                return;   // another caller finished the same download while we waited
            }
            var buffer = new byte[1 << 19];
            var response = await _client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            try
            {
                response.EnsureSuccessStatusCode();
                var supportsRange = response.Headers.AcceptRanges.Contains("bytes");
                totalBytes = response.Content.Headers.ContentLength ?? expectedSize;

                await using var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, buffer.Length, true);
                long received = 0;
                long lastReport = 0;
                bool finished = false;
                int failures = 0;
                Report(0);

                while (!finished)
                {
                    try
                    {
                        await using var content = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                        int read;
                        while ((read = await content.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                        {
                            await fileStream.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                            received += read;
                            var now = Environment.TickCount64;
                            if (now - lastReport >= 500)
                            {
                                lastReport = now;
                                Report(received);
                            }
                        }
                        if (totalBytes is long expected && received < expected)
                        {
                            throw new IOException($"Download of '{fileName}' ended prematurely: received {received} of {expected} bytes.");
                        }
                        finished = true;
                        Report(received);
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException)
                    {
                        if (++failures > 20)
                        {
                            throw;
                        }
                        await Task.Delay(TimeSpan.FromSeconds(Math.Min(30, 2 * failures)), cancellationToken).ConfigureAwait(false);
                        var request = new HttpRequestMessage(HttpMethod.Get, url);
                        if (supportsRange && received > 0)
                        {
                            request.Headers.Range = new RangeHeaderValue(received, null);
                        }
                        else
                        {
                            received = 0;
                            fileStream.Position = 0;
                            fileStream.SetLength(0);
                        }
                        response.Dispose();
                        response = await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                        response.EnsureSuccessStatusCode();
                        if (response.Content.Headers.ContentLength is long len)
                        {
                            totalBytes = received + len;
                        }
                    }
                }
            }
            finally
            {
                response?.Dispose();
            }

            if (expectedSize is long want && new FileInfo(tempPath).Length != want)
            {
                throw new InvalidDataException($"Downloaded '{fileName}' has {new FileInfo(tempPath).Length} bytes, expected {want}.");
            }
            File.Move(tempPath, localPath, overwrite: true);
        }
        catch
        {
            try { File.Delete(tempPath); } catch { /* ignore */ }
            throw;
        }
        finally
        {
            _oneDownloadAtATime.Release();
        }

        void Report(long received)
        {
            if (reportProgress is null)
            {
                return;
            }
            float fraction = totalBytes is long total && total > 0 ? Math.Clamp(received / (float)total, 0f, 1f) : 0f;
            reportProgress(new DownloadProgress(received, totalBytes, fraction, fileName));
        }
    }
}
