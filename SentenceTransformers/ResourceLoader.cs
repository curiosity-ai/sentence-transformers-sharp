/*
 * This file contains a static class named ResourceLoader, which provides utility methods for loading resources from assemblies.
 * It includes methods for opening a resource stream, getting a resource as a byte array, and asynchronously loading a resource using a specified loader function.
 * These methods are useful for accessing embedded resources within an assembly, such as files embedded as resources in a .NET project.
 */
using System.Reflection;

namespace SentenceTransformers
{
    // Provides utility methods for loading resources from assemblies.
    public static class ResourceLoader
    {
        // Opens a resource stream from the specified assembly.
        public static Stream OpenResource(Assembly assembly, string resourceFile)
        {
            // Construct the full resource name.
            string fullResourceName = assembly.GetName().Name + ".Resources." + resourceFile;

            // Return the resource stream.
            return assembly.GetManifestResourceStream(fullResourceName);
        }

        // Gets the resource as a byte array from the specified assembly.
        public static byte[] GetResource(Assembly assembly, string resourceFile)
        {
            // Open the resource stream.
            using Stream resourceStream = OpenResource(assembly, resourceFile);

            // Read the stream into a byte array.
            byte[] buffer = new byte[resourceStream.Length];
            using MemoryStream memoryStream = new MemoryStream(buffer);
            resourceStream.CopyTo(memoryStream);
            return buffer;
        }

        // Asynchronously loads a resource using a specified loader function.
        public static async Task<T> LoadAsync<T>(Assembly assembly, string resourceFile, Func<Stream, Task<T>> loader)
        {
            // Open the resource stream.
            using Stream stream = OpenResource(assembly, resourceFile);

            // Load the resource asynchronously using the provided loader function.
            return await loader(stream);
        }
    }
}
