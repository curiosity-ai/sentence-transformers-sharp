#!/usr/bin/env bash
# Sets up this checkout for local core-library development (macOS / Linux):
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
set -euo pipefail
cd "$(dirname "$0")"

cat > Local.build.props <<'EOF'
<Project>
  <PropertyGroup>
    <UseNuGetSentenceTransformers>false</UseNuGetSentenceTransformers>
  </PropertyGroup>
</Project>
EOF
echo "Wrote Local.build.props (UseNuGetSentenceTransformers=false)"

echo "Deleting bin/ and obj/ folders (stale NuGet assets)..."
find . -type d \( -name bin -o -name obj \) -prune -exec rm -rf {} +

dotnet restore SentenceTransformers.sln
dotnet build SentenceTransformers.sln

echo
echo "Done. Re-open or re-sync the solution in your IDE (Rider: right-click the"
echo "solution -> Restore NuGet Packages, or rebuild) so it picks up the project references."
