# CLAUDE.md

Guidance for Claude Code (and other contributors) when working in this repository.

## Repository layout

- `SentenceTransformers/` — the core library, published to NuGet as the `SentenceTransformers` package (tokenizers, ONNX inference, autograd/LoRA training engine).
- `SentenceTransformers.<Model>/` (MiniLM, ArcticXs, Qwen3, Harrier.Small, Harrier.Medium, Harrier.Small.Pure, Bert.Pure, MiniLMForTest) — per-model wrapper packages, each published as its own NuGet package.
- `SentenceTransformers.Test*/`, `SentenceTransformers.Benchmark*/`, `SentenceTransformers.LoraTraining/` — internal test, benchmark, and training projects (not published).
- `.devops/azure-pipelines.yml` — CI: builds and publishes all packages with a shared CalVer version (`yy.M.<buildId>`).

## Referencing the core SentenceTransformers library

Every consuming project (model packages **and** internal test/benchmark/training projects) references the core library with a single plain project reference:

```xml
<ProjectReference Include="..\SentenceTransformers\SentenceTransformers.csproj" />
```

That is all that is needed, because SDK-style projects do the right thing in both directions:

- **Development / debugging**: everything is a project reference, so the IDE and CLI build core from source and the debugger steps straight into `SentenceTransformers` sources. No setup, no per-machine overrides.
- **Publishing**: when a model package is packed (`dotnet pack`, or `-p:GeneratePackageOnBuild=true` as CI does), the SDK automatically rewrites the `SentenceTransformers` project reference into a **NuGet package dependency** in the produced `.nupkg` — it does *not* bundle core's DLL. The dependency version is core's package version from the same build, i.e. the shared CalVer version CI stamps via `/p:Version=$(targetVersion)` (that property is passed on the command line, so it flows into the referenced core project too). All packages published from one CI run therefore reference each other at one consistent version.

This is the built-in SDK behaviour described in <https://markheath.net/post/multiple-nuget-single-repo>; it replaces the old `UseNuGetSentenceTransformers` package-vs-project switch (and its `Local.build.props`, setup scripts, and stale-restore guard), which existed only to emulate this by hand.

Rules:

- Core (`SentenceTransformers`) must stay packable (it produces `PackageId=SentenceTransformers`). That is what makes the SDK emit a package *dependency* rather than copy the DLL into each model package.
- The whole build graph must be versioned together. CI already does this: one shared CalVer version is stamped across every package in a run. Do not try to pin a model package to a different core version than the rest of the run.

## Building

```bash
dotnet restore SentenceTransformers.sln
dotnet build SentenceTransformers.sln -c Release
```
