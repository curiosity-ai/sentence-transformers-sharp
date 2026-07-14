# Sets up this checkout for local core-library development (Windows):
#   1. writes Local.build.props (UseNuGetSentenceTransformers=false), so the whole
#      solution uses a ProjectReference to SentenceTransformers.csproj and the
#      debugger can step into the core library,
#   2. deletes all bin/ and obj/ folders — IDE "Clean" does NOT delete NuGet's
#      obj/project.assets.json, and stale package-mode assets otherwise clash with
#      the project reference (MSB3243 / CS1704 / CS0006),
#   3. restores and builds the solution.
# Afterwards, re-sync the solution in the IDE (Rider: right-click solution ->
# Restore NuGet Packages, or just rebuild).
# To go back to the NuGet-package default: delete Local.build.props and run the
# clean/restore steps again.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

@'
<Project>
  <PropertyGroup>
    <UseNuGetSentenceTransformers>false</UseNuGetSentenceTransformers>
  </PropertyGroup>
</Project>
'@ | Set-Content -Path Local.build.props -Encoding UTF8
Write-Host "Wrote Local.build.props (UseNuGetSentenceTransformers=false)"

Write-Host "Deleting bin/ and obj/ folders (stale NuGet assets)..."
Get-ChildItem -Path . -Recurse -Directory |
    Where-Object { $_.Name -eq 'bin' -or $_.Name -eq 'obj' } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

dotnet restore SentenceTransformers.sln
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
dotnet build SentenceTransformers.sln
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Re-open or re-sync the solution in your IDE (Rider: right-click the"
Write-Host "solution -> Restore NuGet Packages, or rebuild) so it picks up the project references."
