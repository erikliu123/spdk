#ifndef DDS_H
#define DDS_H


typedef union
{
  struct
  {
    unsigned char r,g,b,a;//b, g, r, a; /*if stay same, decompress image will have problem*/
  };
  unsigned int u;
} Color32;

typedef union
{
  struct
  {
    unsigned short b : 5;
    unsigned short g : 6;
    unsigned short r : 5;
  };
  unsigned short u;
} Color16;
// int func(void);
typedef struct
{
  Color16 col0;
  Color16 col1;
  union
  {
    unsigned char row[4];
    unsigned int indices;
  };
  // void decompress(Color32 colors[16]) const;
} BlockDXT1;

#define MAX_CUDA_PICTURES 10
struct cudaPreAlloc
{
  struct {
    Color32 *compressResult[MAX_CUDA_PICTURES];//1MB
    BlockDXT1 *inputImage[MAX_CUDA_PICTURES];//8MB
    void *d_permutations;
    //BlockDXT1 *inputImageArray;//8MB
    void *devHostDataToDevice;
    uint *permutations;
  } compressTask;

  struct {
    Color32 *decompressResult[MAX_CUDA_PICTURES];//8MB
    BlockDXT1 *inputImage[MAX_CUDA_PICTURES];//1MB
  } decompressTask;

};

/*
    图像压缩/解压缩相关
*/
#if !defined(MAKEFOURCC)
#define MAKEFOURCC(ch0, ch1, ch2, ch3)                \
  ((unsigned int)(ch0) | ((unsigned int)(ch1) << 8) | \
   ((unsigned int)(ch2) << 16) | ((unsigned int)(ch3) << 24))
#endif

typedef unsigned int uint;
typedef unsigned short ushort;

struct DDSPixelFormat {
  uint size;
  uint flags;
  uint fourcc;
  uint bitcount;
  uint rmask;
  uint gmask;
  uint bmask;
  uint amask;
};

struct DDSCaps {
  uint caps1;
  uint caps2;
  uint caps3;
  uint caps4;
};

/// DDS file header.
struct DDSHeader {
  uint fourcc;
  uint size;
  uint flags;
  uint height;
  uint width;
  uint pitch;
  uint depth;
  uint mipmapcount;
  uint reserved[11];
  DDSPixelFormat pf;
  DDSCaps caps;
  uint notused;
};

static const uint FOURCC_DDS = MAKEFOURCC('D', 'D', 'S', ' ');
static const uint FOURCC_DXT1 = MAKEFOURCC('D', 'X', 'T', '1');
static const uint DDSD_WIDTH = 0x00000004U;
static const uint DDSD_HEIGHT = 0x00000002U;
static const uint DDSD_CAPS = 0x00000001U;
static const uint DDSD_PIXELFORMAT = 0x00001000U;
static const uint DDSCAPS_TEXTURE = 0x00001000U;
static const uint DDPF_FOURCC = 0x00000004U;
static const uint DDSD_LINEARSIZE = 0x00080000U;

#endif  // DDS_H
