//文件：test1.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string>


#include "ndp/helper_functions.h"
#include "ndp/helper_cuda.h"

#include "ndp/helper_math.h"
#include "ndp.h"
#include "ndp_cuda.h"

#define ROWS 32
#define COLS 16
#define CHECK(res)        \
  if (res != cudaSuccess) \
  {                       \
    exit(-1);             \
  }


#define spdk_get_ticks(s) gettimeofday(&s, 0)

__global__ void Kerneltest(int **da, unsigned int rows, unsigned int cols)
{
  unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row < rows && col < cols)
  {
    da[row][col] = row * cols + col;
  }
}



cudaPreAlloc gAlloc;

void decompress_simple(BlockDXT1 *dxt, Color32 *colors, int width)
{
  Color32 palette[4];
  Color16 col0 = dxt->col0;
  Color16 col1 = dxt->col1;
  // Does bit expansion before interpolation.
  palette[0].b = (col0.b << 3) | (col0.b >> 2);
  palette[0].g = (col0.g << 2) | (col0.g >> 4);
  palette[0].r = (col0.r << 3) | (col0.r >> 2);
  palette[0].a = 0xFF;

  palette[1].r = (col1.r << 3) | (col1.r >> 2);
  palette[1].g = (col1.g << 2) | (col1.g >> 4);
  palette[1].b = (col1.b << 3) | (col1.b >> 2);
  palette[1].a = 0xFF;

  if (col0.u > col1.u)
  {
    // Four-color block: derive the other two colors.
    palette[2].r = (2 * palette[0].r + palette[1].r) / 3;
    palette[2].g = (2 * palette[0].g + palette[1].g) / 3;
    palette[2].b = (2 * palette[0].b + palette[1].b) / 3;
    palette[2].a = 0xFF;

    palette[3].r = (2 * palette[1].r + palette[0].r) / 3;
    palette[3].g = (2 * palette[1].g + palette[0].g) / 3;
    palette[3].b = (2 * palette[1].b + palette[0].b) / 3;
    palette[3].a = 0xFF;
  }
  else
  {
    // Three-color block: derive the other color.
    palette[2].r = (palette[0].r + palette[1].r) / 2;
    palette[2].g = (palette[0].g + palette[1].g) / 2;
    palette[2].b = (palette[0].b + palette[1].b) / 2;
    palette[2].a = 0xFF;

    palette[3].r = 0x00;
    palette[3].g = 0x00;
    palette[3].b = 0x00;
    palette[3].a = 0x00;
  }

  for (int i = 0; i < 16; i++)
  {
    colors[i/4 * width + i % 4] = palette[(dxt->indices >> (2 * i)) & 0x3];
  }
}

__global__ void 
ndp_accelerate_decompress(BlockDXT1 *input, Color32 *output, int height, int width)
{
  Color32 palette[4];
  //获取当前block应该处理的图像区域
  const int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int tid = bid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  
  if(tid >= (height * width / 16))
    return ;

  BlockDXT1 *dxt = (BlockDXT1 *)input + tid;
  Color16 col0 = dxt->col0;
  Color16 col1 = dxt->col1;  
  int row = 4 * tid / width;
  int col = (4 * tid) % width;
  int start = row * 4 * width + col;
  //printf("%d\n", gridDim.x);
  // Does bit expansion before interpolation.
  palette[0].b = (col0.b << 3) | (col0.b >> 2);
  palette[0].g = (col0.g << 2) | (col0.g >> 4);
  palette[0].r = (col0.r << 3) | (col0.r >> 2);
  palette[0].a = 0xFF;

  palette[1].r = (col1.r << 3) | (col1.r >> 2);
  palette[1].g = (col1.g << 2) | (col1.g >> 4);
  palette[1].b = (col1.b << 3) | (col1.b >> 2);
  palette[1].a = 0xFF;

  if (col0.u > col1.u)
  {
    // Four-color block: derive the other two colors.
    palette[2].r = (2 * palette[0].r + palette[1].r) / 3;
    palette[2].g = (2 * palette[0].g + palette[1].g) / 3;
    palette[2].b = (2 * palette[0].b + palette[1].b) / 3;
    palette[2].a = 0xFF;

    palette[3].r = (2 * palette[1].r + palette[0].r) / 3;
    palette[3].g = (2 * palette[1].g + palette[0].g) / 3;
    palette[3].b = (2 * palette[1].b + palette[0].b) / 3;
    palette[3].a = 0xFF;
  }
  else
  {
    // Three-color block: derive the other color.
    palette[2].r = (palette[0].r + palette[1].r) / 2;
    palette[2].g = (palette[0].g + palette[1].g) / 2;
    palette[2].b = (palette[0].b + palette[1].b) / 2;
    palette[2].a = 0xFF;

    palette[3].r = 0x00;
    palette[3].g = 0x00;
    palette[3].b = 0x00;
    palette[3].a = 0x00;
  }

  for (int i = 0; i < 16; i++)//达到毫秒级别的时延。
  {
      output[tid * 16 + i] = palette[(dxt->indices >> (2 * i)) & 0x3];
    //output[start + i/4 * width + i % 4] = palette[(dxt->indices >> (2 * i)) & 0x3];
  }
}


extern "C"
{
  int ndp_init(void)
  {
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
    checkCudaErrors(cudaMalloc((void **)&(gAlloc.compressTask.inputImage),MAX_COMPRESS_SIZE));
    //checkCudaErrors(cudaHostAlloc((void **)&(gAlloc.compressTask.decompressResult), MAX_COMPRESS_SIZE * 8, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    //checkCudaErrors(cudaHostGetDevicePointer((void **)&gAlloc.compressTask.devHostDataToDevice, gAlloc.compressTask.decompressResult, 0));
    checkCudaErrors(cudaMalloc((void **)&(gAlloc.compressTask.decompressResult), MAX_COMPRESS_SIZE * 8));
    return 0;
  }

  int ndp_free(void)
  {
    checkCudaErrors(cudaFree((void *)gAlloc.compressTask.inputImage));
    checkCudaErrors(cudaFree((void *)gAlloc.compressTask.decompressResult));
    return 0;
  }


  //时间框架
  int ndp_decompress(const char *filename, uint8_t **result, int opt)//选不选择NDP加速
  {
    // Gflops/s
    //统计NDP计算时间
    uint64_t ticks_per_second, start_ticks, end_ticks;
    struct timeval total_begin, total_end;
 
    if (!filename || !result)
      return -ENOENT;

    std::string image_path = DEFAULT_NDP_DIR;
    uint8_t *compressImageBuffer;
    uint8_t *h_result, *tmp_cuda_result;
    image_path += filename;
    gettimeofday(&total_begin, 0);
    image_path = "/home/femu/cuda-samples/Samples/dxtc/data/lena_ref.dds";
    FILE *fp = fopen(image_path.c_str(), "rb");

    if (fp == 0)
    {
      //NVME_NDP_ERRLOG(ctrlr, "Specified timeout would cause integer overflow. Defaulting to no timeout.\n");
      printf("Error, unable to open output image <%s>\n", image_path.c_str());
      return -ENONET;
    }
    //= sdkFindFilePath(filename, argv[0]);
    DDSHeader header;
    // header.notused = 0;
    fread(&header, sizeof(DDSHeader), 1, fp);
    uint w = header.width , h = header.height;
    uint W = w, H = h;

    //根据header.pitch读取文件大小
    compressImageBuffer = (uint8_t *)malloc(header.pitch);
    
    if(compressImageBuffer == nullptr)
    {
      return -ENOMEM;
    }
   
     h_result = (uint8_t *)malloc(w * h * 4);//RGB + alpha 
      if(h_result == nullptr)
      {
        free(compressImageBuffer);
        return -ENOMEM;
      }

    fread(compressImageBuffer, header.pitch, 1, fp);
    fclose(fp);
    uint64_t cuda_start_ticks = 0, cuda_memcpy_ticks = 0, cuda_end_ticks = 0,
    cuda_host_ticks = 0;
    //解压缩操作
    if(opt == 0){
      

      for(int k = 0; k<500; k++)//假设有1000张图片
      for (uint y = 0; y < h; y += 4)
      {
        for (uint x = 0; x < w; x += 4)
        {
          uint referenceBlockIdx = ((y / 4) * (W / 4) + (x / 4));
          uint resultBeginIdx = y * W + x;
          BlockDXT1 *tmp = ((BlockDXT1 *)compressImageBuffer) + referenceBlockIdx;
          Color32 *tmpColor = ((Color32 *)h_result) + resultBeginIdx;
          decompress_simple(tmp, tmpColor, W);
        }
      }
    }
    else if(opt == 1)
    {
        //使用CUDA进行加速
        int BlockSize = 16;
        //图片的长宽都是4的倍数
        int GridSize = (header.width/4 * header.height/4 + (BlockSize*BlockSize - 1)) / (BlockSize*BlockSize);
        dim3 threadPerBlock(BlockSize, BlockSize);
        dim3 numBlocks(GridSize, 1);
        uint64_t temp_start_ticks, temp_end_ticks;
        struct timeval temp_begin, temp_end;
    
        for(int k = 0; k<100; k++){
         
          cudaMemcpy(gAlloc.compressTask.inputImage, compressImageBuffer, header.pitch, cudaMemcpyHostToDevice);
          gettimeofday(&temp_begin, 0);
          ndp_accelerate_decompress<<<numBlocks, threadPerBlock>>>(gAlloc.compressTask.inputImage, (Color32 *)gAlloc.compressTask.decompressResult, H, W);
          checkCudaErrors(cudaDeviceSynchronize());//有问题？
          gettimeofday(&temp_end, 0);
          cudaMemcpy(h_result, gAlloc.compressTask.decompressResult, w * h * 4, cudaMemcpyDeviceToHost);
          printf("NDP consume [%ld] us\n", 1000000*(temp_end.tv_sec-temp_begin.tv_sec) + temp_end.tv_usec-temp_begin.tv_usec);
        }

    }
    gettimeofday(&total_end, 0);

  
    sdkSavePPM4ub("/home/femu/shell/ljh_cuda.ppm", opt ? (unsigned char *)gAlloc.compressTask.devHostDataToDevice:h_result, w, h);
 
    
    *result = h_result;
    free(compressImageBuffer);
    //end_ticks = spdk_get_ticks();
    
    return 0;
  }

  /*读取存储相关的数据后, 执行NDP加速
  */
  //一个完整的NDP命令

  int func(void) // 注意这里定义形式
  {
    int **da = NULL;
    int **ha = NULL;
    int *dc = NULL;
    int *hc = NULL;
    cudaError_t res;
    int r, c;
    bool is_right = true;

    res = cudaMalloc((void **)(&da), ROWS * sizeof(int *));
    CHECK(res)
    res = cudaMalloc((void **)(&dc), ROWS * COLS * sizeof(int));
    CHECK(res)
    ha = (int **)malloc(ROWS * sizeof(int *));
    hc = (int *)malloc(ROWS * COLS * sizeof(int));

    for (r = 0; r < ROWS; r++)
    {
      ha[r] = dc + r * COLS;
    }
    res = cudaMemcpy((void *)(da), (void *)(ha), ROWS * sizeof(int *), cudaMemcpyHostToDevice);
    CHECK(res)
    dim3 dimBlock(16, 16);
    dim3 dimGrid((COLS + dimBlock.x - 1) / (dimBlock.x), (ROWS + dimBlock.y - 1) / (dimBlock.y));
    Kerneltest<<<dimGrid, dimBlock>>>(da, ROWS, COLS);
    res = cudaMemcpy((void *)(hc), (void *)(dc), ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK(res)

    for (r = 0; r < ROWS; r++)
    {
      for (c = 0; c < COLS; c++)
      {
        printf("%4d ", hc[r * COLS + c]);
        if (hc[r * COLS + c] != (r * COLS + c))
        {
          is_right = false;
        }
      }
      printf("\n");
    }
    printf("the result is %s!\n", is_right ? "right" : "false");

    cudaFree((void *)da);
    cudaFree((void *)dc);
    free(ha);
    free(hc);
    //  getchar();
    return 0;
  }
};

int main()
{
    uint8_t *buffer;
    ndp_init();
    ndp_decompress("liu", &buffer, 0);
    free(buffer);
    ndp_decompress("liu", &buffer, 1);
    free(buffer);
    ndp_free();
    return 0;
}
