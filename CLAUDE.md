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

## Internal test/benchmark/training projects

The internal projects (`SentenceTransformers.Test*`, `SentenceTransformers.Benchmark*`, `SentenceTransformers.LoraTraining`) use a `ProjectReference` to the core `SentenceTransformers.csproj` (so local core changes are picked up without publishing) **and** `ProjectReference`s to the model projects. Because the model projects pull the `SentenceTransformers` NuGet package in transitively, every internal project must also carry this line to suppress the package's assets — otherwise the build fails with `CS1704` (two assemblies with the same simple name):

```xml
<ProjectReference Include="..\SentenceTransformers\SentenceTransformers.csproj" />
<PackageReference Include="SentenceTransformers" Version="26.7.2589" ExcludeAssets="all" />
```

The direct `PackageReference` overrides the transitive one from the model projects, and `ExcludeAssets="all"` keeps its compile/runtime assets out of the build, so only the project-built core assembly is used. Keep its `Version` in sync with the pin used by the model packages (same-version rule above). This works because the assemblies are not strong-named, so the runtime binds by simple name and ignores the version difference between the locally built core and the version the model packages were compiled against.

## Building

```bash
dotnet restore SentenceTransformers.sln
dotnet build SentenceTransformers.sln -c Release
```
