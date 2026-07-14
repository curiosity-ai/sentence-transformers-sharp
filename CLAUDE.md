# CLAUDE.md

Guidance for Claude Code (and other contributors) when working in this repository.

## Repository layout

- `SentenceTransformers/` — the core library, published to NuGet as the `SentenceTransformers` package (tokenizers, ONNX inference, autograd/LoRA training engine).
- `SentenceTransformers.<Model>/` (MiniLM, ArcticXs, Qwen3, Harrier.Small, Harrier.Medium, Harrier.Small.Pure, Bert.Pure, MiniLMForTest) — per-model wrapper packages, each published as its own NuGet package.
- `SentenceTransformers.Test*/`, `SentenceTransformers.Benchmark*/`, `SentenceTransformers.LoraTraining/` — internal test, benchmark, and training projects (not published).
- `.devops/azure-pipelines.yml` — CI: builds and publishes all packages with a shared CalVer version (`yy.M.<buildId>`).

## Referencing the core SentenceTransformers library

How projects reference the core library is controlled by the `UseNuGetSentenceTransformers` property, which defaults to `true` in `Directory.Build.props`. Every consuming project (model packages **and** internal test/benchmark/training projects) carries the same conditional pair:

```xml
<PackageReference Include="SentenceTransformers" Version="26.7.2593" Condition="'$(UseNuGetSentenceTransformers)' == 'true'" />
<ProjectReference Include="..\SentenceTransformers\SentenceTransformers.csproj" Condition="'$(UseNuGetSentenceTransformers)' != 'true'" />
```

- **Default (`true`)**: everything references the published `SentenceTransformers` NuGet package. This is what CI uses, so published model `.nupkg`s depend on the pinned package version.
- **Local core development (`false`)**: build with `dotnet build -p:UseNuGetSentenceTransformers=false` (works for `dotnet test`/`run` too) and everything references `SentenceTransformers.csproj` directly, so local core changes are picked up without publishing.

Rules:

- The property must be identical for the whole build graph, which is why it can only be set globally (`Directory.Build.props` default or `-p:` on the command line). Do **not** hardcode it inside an individual `.csproj`: MSBuild properties do not flow into referenced projects, so the model projects would still resolve the package, it would flow transitively into the internal project alongside the project-built core, and the build fails with `CS1704` (two assemblies with the same simple name `SentenceTransformers`).
- Do not replace the conditional pair with a plain `<PackageReference Include="SentenceTransformers" ExcludeAssets="all" />` next to a `ProjectReference`: it compiles, and `dotnet test` even passes (xunit probes the output folder by simple name), but console apps crash at startup with `FileNotFoundException: Could not load file or assembly 'SentenceTransformers, Version=…'` — NuGet unifies the project and the package under one identity, the package node wins, and its excluded assets leave `deps.json` without any runtime entry for `SentenceTransformers.dll`.
- Keep the pinned `SentenceTransformers` package version identical across all `.csproj` files. When bumping, update every project in the same commit (versions are CalVer, published from `.devops/azure-pipelines.yml`; check the latest on nuget.org).
- When core-library changes are needed by a consumer, first merge to `main` (CI publishes a new core package), then bump the pinned `Version` in the consuming projects.

## Building

```bash
dotnet restore SentenceTransformers.sln
dotnet build SentenceTransformers.sln -c Release
```
