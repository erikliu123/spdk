//文件：test1.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>

#include "ndp/helper_functions.h"
#include "ndp/helper_cuda.h"

#include "ndp/helper_math.h"
#include "ndp.h"
#include "ndp_cv.h"
#include "ndp_cuda.h"

extern "C"
{
#include "spdk/env.h"
#include "spdk/log.h"

#include "spdk/bdev.h"
#include "spdk/bdev_module.h"
#include "nvmf_internal.h"
};

#include "file_info.h"
#include <iterator>

#define ROWS 32
#define COLS 16
#define CHECK(res)        \
  if (res != cudaSuccess) \
  {                       \
    exit(-1);             \
  }
extern std::map<std::string, std::pair<int64_t, std::vector<file_extent>>> file_lbas_map;
extern int produce_fsinfo(const char *path, int depth);

static void ndp_wordcount_complete(struct spdk_bdev_io *bdev_io, bool success,
                                   void *cb_arg);

cudaPreAlloc gAlloc;
static int g_subid = 1;
//std::map<int, struct ndp_subrequest *> g_subreq;
std::map<int, std::vector<struct ndp_subrequest *>> search_table; // task id对应的子任务
std::map<int, struct ndp_request *> ndp_wait_queue; 

void spdk_ndp_request_complete(struct ndp_subrequest *sub_req, int status)
{
  struct ndp_request *ndp_req = sub_req->req;
  ndp_req->num_finished_jobs++;
  if (ndp_req->spdk_read_flag)
  {
    spdk_dma_free(sub_req->read_ptr);
  }
  else
    free(sub_req->read_ptr);
  if (ndp_req->num_finished_jobs == ndp_req->num_jobs && ndp_req->spdk_read_flag)//异步方式
  {
    spdk_nvmf_request_complete(ndp_req->nvmf_req);
    // TODO:打印信息？
    free(ndp_req);
  }
}


extern "C" {
int register_ndp_task(struct ndp_request *ndp_req)
{ 
 // std::vector<struct ndp_subrequest *> empty;
  if(ndp_wait_queue.count(ndp_req->id)){
    
    return -1;
  }
  ndp_wait_queue[ndp_req->id] = ndp_req;
  return 0;
}

struct ndp_request* get_ndp_req_from_id(int id)
{
  if(ndp_wait_queue.count(id))
  {
    return ndp_wait_queue[id];
  }
  return NULL;
}

//分配和启动子请求
int alloc_and_start_sub_req(struct ndp_request *ndp_req)
{
  int len = strlen(ndp_req->read_path);
  int ret = 0;
  std::pair<int64_t, std::vector<file_extent>> lba;
  ndp_req->num_finished_jobs = ndp_req->num_jobs = 0;
  
  if (ndp_req->reverse)
  { // TODO
    printf("unsupported REVERSE...\n");
    return -1;
  }
  else if (ndp_req->dir_flag)
  {
    //(ndp_req, input_name, ndp_face_detection_complete);
    switch (ndp_req->task_type)
    {
    case FACE_DETECTION:
      //查找所有的bmp文件
      {
        std::map<std::string, std::pair<int64_t, std::vector<file_extent>>>::iterator it = file_lbas_map.lower_bound(ndp_req->read_path);
        while (it != file_lbas_map.end() && strncmp(it->first.c_str(), ndp_req->read_path, len) == 0)
        {
          //文件前缀和low_bound一样时
          int sublen = it->first.size();
          if (it->second.first > 0 && it->first.substr(sublen - 4) == ".bmp")//最后三个字符
          {
            ndp_req->num_jobs++;
            std::cout << it->first << std::endl;
            //生成子请求并记录
            struct ndp_subrequest *sub_req = (struct ndp_subrequest *)malloc(sizeof(struct ndp_subrequest));
            strcpy(sub_req->read_file, it->first.c_str());
            sub_req->req = ndp_req;
            sub_req->read_ptr = NULL;
            //红黑树和链表管理。。。
            search_table[ndp_req->id].push_back(sub_req);
          }
          it++;
        }
      }

      break;
    case COMPRESS:
      break;
    default:
      break;
    }
  }
  else
  {
    if(len > 0){//传递了一个目录文件过来
      struct ndp_subrequest *sub_req = (struct ndp_subrequest *)malloc(sizeof(struct ndp_subrequest));
      strcpy(sub_req->read_file, ndp_req->read_path);
      sub_req->req = ndp_req;
      ndp_req->num_jobs++;
      sub_req->read_ptr = NULL;
            //红黑树和链表管理。。。
      search_table[ndp_req->id].push_back(sub_req);
    }
    //不需要子任务
  }

  //之前的文件合法性已经检查完毕，不考虑极端情况下文件系统同时被篡改。。
  for (auto sub_req : search_table[ndp_req->id])
  {
    switch (ndp_req->task_type)
    {
    case FACE_DETECTION:
    {
      ret = ndp_read_file(sub_req, sub_req->read_file, ndp_face_detection_complete, sub_req);

      if (!ret && !ndp_req->spdk_read_flag)
      {
        printf("direct IO ...\n");
        ret = ndp_compute_face_detection(sub_req);
      }
    }
    break;
    
    default:
      return -1;
      break;
    }
  }
  // if (!ndp_req->spdk_read_flag)
  // {
  //   spdk_nvmf_request_complete(ndp_req->nvmf_req);
  //   free(ndp_req);
  // }
  return 0;
}

};

//读取文件的统一接口
int ndp_read_file(struct ndp_subrequest *sub_req, char *input_name, void (*callback_fn)(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg), void *arg)
{
  std::pair<int64_t, std::vector<file_extent>> lba;
  struct ndp_request *ndp_req = sub_req->req;
  bool spdk_read_flag = ndp_req->spdk_read_flag;
  struct spdk_nvme_cpl *response = &ndp_req->nvmf_req->rsp->nvme_cpl;

  response->cdw0 = 0;
  response->status.sc = 0;
  response->status.sct = 0;
  if (file_lbas_map.find(input_name) != file_lbas_map.end()) //获取文件大小、元数据
    lba = file_lbas_map[input_name];
  else
    return -ENOENT;
  if (lba.first == 0) // directory or empty file
    return -ENOENT;

  if (spdk_read_flag) // SPDK读取数据
  {
    //在mnt/ndp下读取数据
    //得到IO时延，必须是direct IO
    /*读取数据，离散的更好*/
    sub_req->malloc_time = spdk_get_ticks();
    unsigned char *readbuf = (unsigned char *)spdk_dma_zmalloc(lba.first + BLK_SIZE, 0x200000, NULL);
    assert(readbuf != NULL);
    sub_req->read_bdev_blocks = 0;
    sub_req->total_bdev_blocks = lba.second.size();
    sub_req->read_ptr = readbuf;
    sub_req->total_len = lba.first;
    // spdk_bdev_open_ext(BDEV_NAME, true, NULL, NULL,&desc);
    // SPDK_NOTICELOG("OPEN DEVICE SUUCEESFULLY");
    // io_ch = spdk_bdev_get_io_channel(desc);
    int offset = 0;
    sub_req->start_io_time = spdk_get_ticks();
    for (int i = 0; i < lba.second.size(); i++)
    {
      spdk_bdev_read(ndp_req->desc, ndp_req->io_ch, readbuf + offset, lba.second[i].first_block << FILE_SYS_BIT, lba.second[i].block_count << BLK_SHIFT_BIT, callback_fn, arg);
      // spdk_bdev_read(ndp_req->desc, ndp_req->io_ch, readbuf + offset, 0x400 , 512, ndp_face_detection_complete, ndp_req);
      offset += lba.second[i].block_count << BLK_SHIFT_BIT;
      SPDK_INFOLOG(ndp, "read lba[%lld], len[%lld]\n", lba.second[i].first_block, lba.second[i].block_count << BLK_SHIFT_BIT);
    }
  }
  else //同步IO
  {
    int fd, cnt;
    fd = open(input_name, O_DIRECT | O_RDONLY);
    if (fd < 0)
      return -ENOENT;
    int blk_size = getpagesize();
    int readlen = (lba.first + blk_size - 1) / blk_size * blk_size;
    unsigned char *tempbuf;
    sub_req->malloc_time = spdk_get_ticks();
    posix_memalign((void **)&tempbuf, blk_size, readlen);
    sub_req->read_ptr = tempbuf;

    sub_req->start_io_time = spdk_get_ticks();
    cnt = read(fd, tempbuf, readlen);
    sub_req->end_io_time = spdk_get_ticks();
    assert(cnt > 0);
    //callback_fn(ndp_req->arg);
    // SPDK_INFOLOG(ndp, "malloc time: %.2f us, diret IO time: %.2f us.\n", 1000000.0 * (ndp_req->start_io_time - ndp_req->malloc_time) / spdk_get_ticks_hz(), 1000000.0 * (spdk_get_ticks() - ndp_req->start_io_time) / spdk_get_ticks_hz());
  }
  return 0;
}

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
    colors[i / 4 * width + i % 4] = palette[(dxt->indices >> (2 * i)) & 0x3];
  }
}

__global__ void
ndp_accelerate_decompress(BlockDXT1 *input, Color32 *output, int height, int width)
{
  Color32 palette[4];
  //获取当前block应该处理的图像区域
  const int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int tid = bid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  if (tid >= (height * width / 16))
    return;

  BlockDXT1 *dxt = (BlockDXT1 *)input + tid;
  Color16 col0 = dxt->col0;
  Color16 col1 = dxt->col1;
  int row = 4 * tid / width;
  int col = (4 * tid) % width;
  int start = row * 4 * width + col;
  // printf("%d\n", gridDim.x);
  //  Does bit expansion before interpolation.
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
    output[start + i / 4 * width + i % 4] = palette[(dxt->indices >> (2 * i)) & 0x3];
  }
}

extern "C"
{
  int ndp_init(void)
  {
    produce_fsinfo(DEFAULT_NDP_DIR, 0); //加载系统的文件信息
    checkCudaErrors(cudaMalloc((void **)&(gAlloc.compressTask.inputImage), MAX_COMPRESS_SIZE));
#ifdef ZERO_COPY
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
    checkCudaErrors(cudaHostAlloc((void **)&(gAlloc.compressTask.decompressResult), MAX_COMPRESS_SIZE * 8, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&gAlloc.compressTask.devHostDataToDevice, gAlloc.compressTask.decompressResult, 0));
#else
    checkCudaErrors(cudaMalloc((void **)&(gAlloc.compressTask.decompressResult), MAX_COMPRESS_SIZE * 8));
#endif
    spdk_log_set_flag("ndp");
    return 0;
  }

  int ndp_free(void)
  {
    checkCudaErrors(cudaFree((void *)gAlloc.compressTask.inputImage));
    checkCudaErrors(cudaFree((void *)gAlloc.compressTask.decompressResult));
    return 0;
  }

  int ndp_wordcount(struct ndp_subrequest *sub_req, const char *input_name)
  {
    //单词和数目
    struct ndp_request *ndp_req = sub_req->req;
    std::pair<int64_t, std::vector<file_extent>> lba;
    if (file_lbas_map.find(input_name) != file_lbas_map.end()) //获取文件大小、元数据
      lba = file_lbas_map[input_name];
    else
    {
      return -ENOENT;
    }

    unsigned char *readbuf = (unsigned char *)spdk_dma_zmalloc(lba.first, 0x200000, NULL); //如果是单纯的读操作，则不需要分配空间
    assert(readbuf != NULL);
    //  char *readbuf = (char *)malloc(lba.first + BLKSIZE);
    sub_req->read_bdev_blocks = 0;
    sub_req->total_bdev_blocks = lba.second.size();
    sub_req->read_ptr = readbuf;
    sub_req->total_len = lba.first;
    sub_req->start_io_time = spdk_get_ticks();
    int offset = 0;
    for (int i = 0; i < lba.second.size(); i++)
    {
      spdk_bdev_read(ndp_req->desc, ndp_req->io_ch, readbuf + offset, lba.second[i].first_block << FILE_SYS_BIT, lba.second[i].block_count << BLK_SHIFT_BIT, ndp_wordcount_complete, sub_req);
      // spdk_bdev_read(ndp_req->desc, ndp_req->io_ch, readbuf + offset, 0x400 , 512, ndp_face_detection_complete, ndp_req);
      offset += lba.second[i].block_count << BLK_SHIFT_BIT;
      SPDK_NOTICELOG("read lba[%lld], len[%lld]\n", lba.second[i].first_block, lba.second[i].block_count << BLK_SHIFT_BIT);
    }
    return 0;
  }
  //时间框架
  int ndp_decompress(const char *filename, uint8_t **result, int opt) //选不选择NDP加速
  {
    // Gflops/s
    //统计NDP计算时间
    uint64_t ticks_per_second, start_ticks, end_ticks;
    spdk_log_set_flag("ndp");
    ticks_per_second = spdk_get_ticks_hz();
    start_ticks = spdk_get_ticks();

    if (!filename || !result)
      return -ENOENT;

    std::string image_path = DEFAULT_NDP_DIR;
    uint8_t *compressImageBuffer;
    uint8_t *h_result;
    image_path += filename;

    image_path = "/home/femu/cuda-samples/Samples/dxtc/data/lena_ref.dds";
    FILE *fp = fopen(image_path.c_str(), "rb");

    if (fp == 0)
    {
      // NVME_NDP_ERRLOG(ctrlr, "Specified timeout would cause integer overflow. Defaulting to no timeout.\n");
      printf("Error, unable to open output image <%s>\n", image_path.c_str());
      return -ENONET;
    }
    //= sdkFindFilePath(filename, argv[0]);
    DDSHeader header;
    // header.notused = 0;
    fread(&header, sizeof(DDSHeader), 1, fp);
    uint w = header.width, h = header.height;
    uint W = w, H = h;

    //根据header.pitch读取文件大小
    compressImageBuffer = (uint8_t *)malloc(header.pitch);

    if (compressImageBuffer == nullptr)
    {
      return -ENOMEM;
    }
    h_result = (uint8_t *)malloc(w * h * 4); // RGB + alpha
    if (h_result == nullptr)
    {
      free(compressImageBuffer);
      return -ENOMEM;
    }

    fread(compressImageBuffer, header.pitch, 1, fp);
    fclose(fp);
    uint64_t cuda_start_ticks = 0, cuda_memcpy_ticks = 0, cuda_end_ticks = 0,
             cuda_host_ticks = 0;
    //解压缩操作
    if (opt == 0)
    {
      cuda_memcpy_ticks = spdk_get_ticks();
      for (int k = 0; k < 10; k++) //假设有500张图片
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
      cuda_end_ticks = spdk_get_ticks();
    }
    else if (opt == 1)
    {
      //使用CUDA进行加速
      int BlockSize = 16;
      //图片的长宽都是4的倍数
      int GridSize = (header.width / 4 * header.height / 4 + (BlockSize * BlockSize - 1)) / (BlockSize * BlockSize);
      dim3 threadPerBlock(BlockSize, BlockSize);
      dim3 numBlocks(GridSize, 1);
      uint64_t temp_start_ticks, temp_end_ticks;
      cuda_start_ticks = spdk_get_ticks();
      for (int k = 0; k < 10; k++)
      {
        temp_start_ticks = spdk_get_ticks();
        cudaMemcpy(gAlloc.compressTask.inputImage, compressImageBuffer, header.pitch, cudaMemcpyHostToDevice);
        cuda_memcpy_ticks = spdk_get_ticks();
        if ((1000 * (cuda_memcpy_ticks - temp_start_ticks) / ticks_per_second) > 1)
        {
          SPDK_INFOLOG(ndp, "[%d] NDP's memcpy time too high... %ld (us)\n", k, 1000000 * (cuda_memcpy_ticks - temp_start_ticks) / ticks_per_second);
        }
        ndp_accelerate_decompress<<<numBlocks, threadPerBlock>>>(gAlloc.compressTask.inputImage, gAlloc.compressTask.decompressResult, H, W);
        checkCudaErrors(cudaDeviceSynchronize());

        temp_end_ticks = spdk_get_ticks();
        cudaMemcpy(h_result, gAlloc.compressTask.decompressResult, w * h * 4, cudaMemcpyDeviceToHost);
        if ((1000 * 1000 * (temp_end_ticks - cuda_memcpy_ticks) / ticks_per_second) > 50)
        {
          SPDK_INFOLOG(ndp, "[%d] NDP's kernel time too high... %ld (us)\n", k, 1000000 * (temp_end_ticks - cuda_memcpy_ticks) / ticks_per_second);
        }
      }
    }

#ifdef ZERO_COPY
    sdkSavePPM4ub("/home/femu/shell/ljh_cuda.ppm", opt ? (unsigned char *)gAlloc.compressTask.devHostDataToDevice : h_result, w, h);
#else
    sdkSavePPM4ub("/home/femu/shell/ljh_cuda.ppm", h_result, w, h);
#endif

    *result = h_result;
    free(compressImageBuffer);
    end_ticks = spdk_get_ticks();
    SPDK_INFOLOG(ndp, "MODULE NDP's total process time: %ld (us)\n", 1000 * 1000 * (end_ticks - start_ticks) / ticks_per_second);
    if (opt == 1)
      SPDK_INFOLOG(ndp, "NDP's memcpy1 time: %ld (us), kernel function time:  %ld (us) answer memcpy %ld (us)\n", 1000 * 1000 * (cuda_memcpy_ticks - cuda_start_ticks) / ticks_per_second, 1000 * 1000 * (cuda_end_ticks - cuda_memcpy_ticks) / ticks_per_second, 1000 * 1000 * (cuda_host_ticks - cuda_end_ticks) / ticks_per_second);
    else
      SPDK_INFOLOG(ndp, "kernel function time:  %ld\n", 1000 * 1000 * (cuda_end_ticks - cuda_memcpy_ticks) / ticks_per_second);
    // printf("total cosume time: %ld\n", (end_ticks - start_ticks)/ticks_per_ms);

    return 0;
  }
};

/* 读取存储相关的数据后, 执行NDP加速 */

static void ndp_wordcount_complete(struct spdk_bdev_io *bdev_io, bool success,
                                   void *cb_arg)
{
  struct ndp_subrequest *sub_req = (struct ndp_subrequest *)cb_arg;
  uint64_t ticks_per_second;
  //struct spdk_nvme_cpl *response = &sub_req->req->nvmf_req->rsp->nvme_cpl;
  int sc = 0, sct = 0;
  uint32_t cdw0 = 0;

  //cnn_flag = ndp_req->accel;                               //是否使用加速场景, cdw13
  spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc); //假设中途没有数据块读取错误
  SPDK_INFOLOG(ndp, "cdw0=%d sct=%d, sc=%d\n", cdw0, sct, sc);

  //统计读取的数据是否完成，完成才完成请求

  sub_req->read_bdev_blocks++;
  spdk_bdev_free_io(bdev_io);
  if (sub_req->read_bdev_blocks == sub_req->total_bdev_blocks)
  {
    sub_req->end_io_time = spdk_get_ticks();
    ticks_per_second = spdk_get_ticks_hz();
    SPDK_INFOLOG(ndp, "IO consume: %ld ticks, per seconed ticks: %ld, HEADER[%x]\n", sub_req->end_io_time - sub_req->start_io_time, ticks_per_second, sub_req->read_ptr[0]);
    //是否可以多线程同时处理

    //统计单词数目
    std::map<std::string, int> result;
  }
}
