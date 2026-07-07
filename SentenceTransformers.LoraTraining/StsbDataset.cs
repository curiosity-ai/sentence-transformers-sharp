using System.Globalization;
using System.Text;
using SentenceTransformers.Training;

namespace SentenceTransformers.LoraTraining;

/// <summary>
/// Downloader and parser for the English <b>STS Benchmark</b> (STS-B) dataset — a standard collection
/// of sentence pairs annotated with a human similarity score from 0 (unrelated) to 5 (equivalent). It
/// is a natural fit for testing adapter fine-tuning: the high-score pairs act as related "positive"
/// pairs for contrastive training, while the full graded set drives the STS Spearman evaluation.
///
/// <para>Data is fetched from the permissively licensed <c>stsb_multi_mt</c> mirror on GitHub
/// (<c>PhilipMay/stsb-multi-mt</c>), which republishes the original STS-B splits as simple CSV files.</para>
/// </summary>
internal static class StsbDataset
{
    public const string TrainUrl = "https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/main/data/stsb-en-train.csv";
    public const string DevUrl   = "https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/main/data/stsb-en-dev.csv";
    public const string TestUrl  = "https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/main/data/stsb-en-test.csv";

    public static string TrainPath(string dir) => Path.Combine(dir, "stsb-en-train.csv");
    public static string DevPath(string dir)   => Path.Combine(dir, "stsb-en-dev.csv");
    public static string TestPath(string dir)  => Path.Combine(dir, "stsb-en-test.csv");

    /// <summary>Downloads the train / dev / test splits into <paramref name="dir"/> (skipping files already present).</summary>
    public static async Task DownloadAsync(string dir, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(dir);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };

        await DownloadOneAsync(http, TrainUrl, TrainPath(dir), cancellationToken);
        await DownloadOneAsync(http, DevUrl,   DevPath(dir),   cancellationToken);
        await DownloadOneAsync(http, TestUrl,  TestPath(dir),  cancellationToken);
    }

    private static async Task DownloadOneAsync(HttpClient http, string url, string path, CancellationToken cancellationToken)
    {
        if (File.Exists(path) && new FileInfo(path).Length > 0)
        {
            Console.WriteLine($"  already present: {Path.GetFileName(path)}");
            return;
        }

        Console.WriteLine($"  downloading {Path.GetFileName(path)} ...");
        var bytes = await http.GetByteArrayAsync(url, cancellationToken);
        await File.WriteAllBytesAsync(path, bytes, cancellationToken);
        Console.WriteLine($"    saved {bytes.Length:N0} bytes");
    }

    /// <summary>
    /// Loads a STS-B CSV split as a <see cref="SentencePairDataset"/>. Each row is
    /// <c>sentence1,sentence2,score</c> with the score rescaled from the raw 0–5 range into <c>[0,1]</c>.
    /// </summary>
    public static SentencePairDataset Load(string path)
    {
        var pairs = new List<SentencePair>();

        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var fields = ParseCsvLine(line);
            if (fields.Count < 3) continue;

            string a = fields[0];
            string b = fields[1];

            if (!float.TryParse(fields[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float raw))
            {
                continue; // skip a header row or malformed line
            }

            float score = Math.Clamp(raw / 5f, 0f, 1f);
            pairs.Add(new SentencePair(a, b, score));
        }

        if (pairs.Count == 0)
        {
            throw new InvalidDataException($"No pairs parsed from '{path}'. Is it a valid STS-B CSV?");
        }

        return new SentencePairDataset(pairs);
    }

    // Minimal RFC-4180-ish CSV field parser: handles quoted fields and doubled quotes.
    private static List<string> ParseCsvLine(string line)
    {
        var fields  = new List<string>();
        var sb      = new StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];

            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else                                             { inQuotes = false; }
                }
                else
                {
                    sb.Append(c);
                }
            }
            else
            {
                switch (c)
                {
                    case '"':  inQuotes = true; break;
                    case ',':  fields.Add(sb.ToString()); sb.Clear(); break;
                    default:   sb.Append(c); break;
                }
            }
        }

        fields.Add(sb.ToString());
        return fields;
    }
}
