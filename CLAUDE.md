# CLAUDE.md

Guidance for Claude Code (and other contributors) when working in this repository.

## Repository layout

- `SentenceTransformers/` — the core library, published to NuGet as the `SentenceTransformers` package (tokenizers, ONNX inference, autograd/LoRA training engine).
- `SentenceTransformers.<Model>/` (MiniLM, ArcticXs, Qwen3, Harrier.Small, Harrier.Medium, Harrier.Small.Pure, Bert.Pure, MiniLMForTest) — per-model wrapper packages, each published as its own NuGet package.
- `SentenceTransformers.Test*/`, `SentenceTransformers.Benchmark*/`, `SentenceTransformers.LoraTraining/` — internal test, benchmark, and training projects (not published).
- `.devops/azure-pipelines.yml` — CI: builds and publishes all packages with a shared CalVer version (`yy.M.<buildId>`).

## Referencing the core SentenceTransformers library

**Always use a NuGet `PackageReference` to the published `SentenceTransformers` package — never a `ProjectReference` to `SentenceTransformers.csproj` — in every consuming project that is published:**

```xml
<PackageReference Include="SentenceTransformers" Version="26.7.2589" />
```

Do **not** add:

```xml
<!-- WRONG — do not use -->
<ProjectReference Include="..\SentenceTransformers\SentenceTransformers.csproj" />
```

Rules:

- This applies to all model packages, but not to the internal test/benchmark/training projects. 
- Keep the `SentenceTransformers` package version identical across all `.csproj` files. When bumping, update every project in the same commit (versions are CalVer, published from `.devops/azure-pipelines.yml`; check the latest on nuget.org).
- When core-library changes are needed by a consumer, first merge and publish the core `SentenceTransformers` package (CI on `main` publishes it), then bump the pinned `Version` in the consuming projects.
