// Reference build of the image decode/resize used by LiteRT-LM (stb_image + stb_image_resize2),
// for scripts/generate_embeddinggemma2_reference.py:
//   curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
//   curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h
//   gcc -O2 -shared -fPIC -o libstbref.so stbref.c -lm
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include <string.h>
// Decode to RGB (3 channels). Returns malloc'd pixels; caller frees with stbref_free.
unsigned char* stbref_load(const unsigned char* data, int len, int* w, int* h) {
  int c; return stbi_load_from_memory(data, len, w, h, &c, 3);
}
void stbref_free(void* p) { stbi_image_free(p); }
int stbref_resize(const unsigned char* in, int w, int h, unsigned char* out, int nw, int nh) {
  return stbir_resize(in, w, h, 0, out, nw, nh, 0, STBIR_RGB, STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM) != 0;
}
