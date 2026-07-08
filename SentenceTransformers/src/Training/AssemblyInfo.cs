using System.Runtime.CompilerServices;

// Exposes internal training hooks (e.g. the numerical-gradient test entry point) to the test project.
[assembly: InternalsVisibleTo("SentenceTransformers.Tests")]
