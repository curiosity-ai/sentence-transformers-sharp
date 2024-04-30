using System.Reflection;

namespace MiniLM.Shared;

public static class ResourceLoader
{
    public static Stream OpenResource(Assembly assembly, string resourceFile)
    {
        return assembly.GetManifestResourceStream(assembly.GetName().Name + ".Resources." + resourceFile);
    }

    public static byte[] GetResource(Assembly assembly, string resourceFile)
    {
        var s  = assembly.GetManifestResourceStream(assembly.GetName().Name + ".Resources." + resourceFile);
        var b  = new byte[s.Length];
        var ms = new MemoryStream(b);
        s.CopyTo(ms);
        return b;
    }

    public static async Task<T> LoadAsync<T>(Assembly assembly, string resourceFile, Func<Stream, Task<T>> loader)
    {
        using Stream stream = OpenResource(assembly, resourceFile);
        return await loader(stream);
    }
}