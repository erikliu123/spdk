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
#include "ndp/dxt.h"
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

/* 普通算子 */
static void ndp_wordcount_complete(struct spdk_bdev_io *bdev_io, bool success,
                                   void *cb_arg);

void decompress_simple(BlockDXT1 *dxt, Color32 *colors, int width);
static int ndp_compute_compress(struct ndp_subrequest *sub_req);

/* 加速算子 */
__global__ void
ndp_accelerate_decompress(BlockDXT1 *input, Color32 *output, int height, int width);

static void computePermutations(uint *permutations);

cudaPreAlloc gAlloc;
static int g_subid = 1;
// std::map<int, struct ndp_subrequest *> g_subreq;
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
  if (ndp_req->num_finished_jobs == ndp_req->num_jobs && ndp_req->spdk_read_flag) //异步方式
  {
    ndp_req->end_time = spdk_get_ticks(); // ndp_req->start_time= spdk_get_ticks();
    spdk_log_set_flag("ndp");
    SPDK_INFOLOG(ndp, "total jobs %d, total task excution time [%lu], IO time[%lu]\n", ndp_req->num_jobs, (ndp_req->end_time - ndp_req->start_time) / 3500, ndp_req->total_io_time);
    spdk_nvmf_request_complete(ndp_req->nvmf_req);
    spdk_log_clear_flag("ndp");
    for (auto x : search_table[ndp_req->id])
    {
      free(x);
    }
    free(ndp_req);
  }
}

int ndp_compute_decompress(struct ndp_subrequest *sub_req)
{
  struct ndp_request *ndp_req = sub_req->req;
  struct spdk_nvmf_request *req = ndp_req->nvmf_req;
  struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl; //可能会因为超时被释放掉
  bool cnn_flag = false;
  int sc = 0, sct = 0;
  int ret = 0;
  uint32_t cdw0 = 0;
  unsigned char *compressImageBuffer = (unsigned char *)sub_req->read_ptr + sizeof(DDSHeader);

  void *h_result = NULL;
  // Color32 palette[4];
  // //获取当前block应该处理的图像区域
  // const int bid = blockIdx.x + blockIdx.y * gridDim.x;
  // const int tid = bid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  DDSHeader header;
  // header.notused = 0;
  memcpy(&header, sub_req->read_ptr, sizeof(DDSHeader));
  uint w = header.width, h = header.height;
  uint W = w, H = h;
  sub_req->result_ptr = (unsigned char *)malloc(8 * header.pitch);
  h_result = sub_req->result_ptr;
  if (!ndp_req->accel)
  { //不加速
    for (uint y = 0; y < h; y += 4)
    {
      for (uint x = 0; x < w; x += 4)
      {
        uint referenceBlockIdx = ((y / 4) * (W / 4) + (x / 4));
        uint resultBeginIdx = y * W + x;
        BlockDXT1 *tmp = ((BlockDXT1 *)compressImageBuffer) + referenceBlockIdx;
        Color32 *tmpColor = ((Color32 *)sub_req->result_ptr) + resultBeginIdx;
        decompress_simple(tmp, tmpColor, W);
      }
    }
  }
  else
  {
    //使用CUDA进行加速
    int BlockSize = 16;
    //图片的长宽都是4的倍数
    int GridSize = (header.width / 4 * header.height / 4 + (BlockSize * BlockSize - 1)) / (BlockSize * BlockSize);
    dim3 threadPerBlock(BlockSize, BlockSize);
    dim3 numBlocks(GridSize, 1);
    uint64_t cuda_start_ticks = 0, cuda_memcpy_ticks = 0, cuda_end_ticks = 0,
             cuda_host_ticks = 0;
    uint64_t ticks_per_second = spdk_get_ticks_hz();
    ;
    uint64_t temp_start_ticks, temp_end_ticks;

    temp_start_ticks = spdk_get_ticks();
    cudaMemcpy(gAlloc.decompressTask.inputImage[sub_req->sub_id], compressImageBuffer, header.pitch, cudaMemcpyHostToDevice);

    cuda_memcpy_ticks = spdk_get_ticks();
    if ((1000 * (cuda_memcpy_ticks - temp_start_ticks) / ticks_per_second) > 1)
    {
      SPDK_INFOLOG(ndp, "!!! NDP's memcpy time too high... %ld (us)\n", 1000000 * (cuda_memcpy_ticks - temp_start_ticks) / ticks_per_second);
    }
    ndp_accelerate_decompress<<<numBlocks, threadPerBlock>>>(gAlloc.decompressTask.inputImage[sub_req->sub_id], gAlloc.decompressTask.decompressResult[sub_req->sub_id], H, W);
    checkCudaErrors(cudaDeviceSynchronize());

    temp_end_ticks = spdk_get_ticks();
    cudaMemcpy(h_result, gAlloc.decompressTask.decompressResult[sub_req->sub_id], w * h * 4, cudaMemcpyDeviceToHost);
    if ((1000 * 1000 * (temp_end_ticks - cuda_memcpy_ticks) / ticks_per_second) > 50)
    {
      SPDK_INFOLOG(ndp, "!!! NDP's kernel time too high... %ld (us)\n", 1000000 * (temp_end_ticks - cuda_memcpy_ticks) / ticks_per_second);
    }
  }

  sub_req->end_compute_time = spdk_get_ticks(); //计算结束时间
  //打印时间戳
  SPDK_INFOLOG(ndp, "file[%s], exclude malloc, total time: %lu, SPDK malloc time: %lu , SPDK IO consume: %lu, computation time: %lu\n", sub_req->read_file, (sub_req->end_compute_time - sub_req->start_io_time) / 3500, (sub_req->start_io_time - sub_req->malloc_time) / 3500, (sub_req->end_io_time - sub_req->start_io_time) / 3500, (sub_req->end_compute_time - sub_req->end_io_time) / 3500);
  return 0;
}

void ndp_compress_complete(struct spdk_bdev_io *bdev_io, bool success,
                           void *cb_arg)
{

  struct ndp_subrequest *sub_req = (struct ndp_subrequest *)cb_arg;
  struct ndp_request *ndp_req = sub_req->req;
  struct spdk_nvmf_request *req = ndp_req->nvmf_req;
  int sc = 0, sct = 0;
  int ret = 0;
  uint32_t cdw0 = 0;
  // spdk_log_set_flag("ndp");
  spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc); //假设中途没有数据块读取错误
  // SPDK_INFOLOG(ndp, "cdw0=%d sct=%d, sc=%d\n", cdw0, sct, sc);

  sub_req->read_bdev_blocks++;
  spdk_bdev_free_io(bdev_io);
  SPDK_INFOLOG(ndp, "continuous blk: %d, total read blocks:%d\n", sub_req->read_bdev_blocks, sub_req->total_bdev_blocks);
  if (sub_req->read_bdev_blocks == sub_req->total_bdev_blocks)
  {
    //读取人脸
    sub_req->end_io_time = spdk_get_ticks();
    ndp_req->total_io_time += (sub_req->end_io_time - sub_req->start_io_time) / 3500;
    ret = ndp_compute_compress(sub_req);
    spdk_ndp_request_complete(sub_req, ret);
  }
}

void ndp_decompress_complete(struct spdk_bdev_io *bdev_io, bool success,
                             void *cb_arg)
{

  struct ndp_subrequest *sub_req = (struct ndp_subrequest *)cb_arg;
  struct ndp_request *ndp_req = sub_req->req;
  struct spdk_nvmf_request *req = ndp_req->nvmf_req;
  struct spdk_nvme_cpl *response = &req->rsp->nvme_cpl; //可能会因为超时被释放掉
  bool cnn_flag = false;
  int sc = 0, sct = 0;
  int ret = 0;
  uint32_t cdw0 = 0;
  spdk_bdev_io_get_nvme_status(bdev_io, &cdw0, &sct, &sc); //假设中途没有数据块读取错误
  SPDK_INFOLOG(ndp, "cdw0=%d sct=%d, sc=%d\n", cdw0, sct, sc);
  //统计读取的数据是否完成，完成才完成请求
  sub_req->read_bdev_blocks++;
  spdk_bdev_free_io(bdev_io);
  SPDK_INFOLOG(ndp, "continuous blk: %d, total read blocks:%d\n", sub_req->read_bdev_blocks, sub_req->total_bdev_blocks);
  if (sub_req->read_bdev_blocks == sub_req->total_bdev_blocks)
  {
    //读取人脸
    sub_req->end_io_time = spdk_get_ticks();
    ndp_req->total_io_time += (sub_req->end_io_time - sub_req->start_io_time) / 3500;
    ret = ndp_compute_decompress(sub_req);
    spdk_ndp_request_complete(sub_req, ret);
  }
}

extern "C"
{
  int register_ndp_task(struct ndp_request *ndp_req)
  {
    // std::vector<struct ndp_subrequest *> empty;
    if (ndp_wait_queue.count(ndp_req->id))
    {

      return -1;
    }
    ndp_wait_queue[ndp_req->id] = ndp_req;
    return 0;
  }

  struct ndp_request *get_ndp_req_from_id(int id)
  {
    if (ndp_wait_queue.count(id))
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
    int subID = 0;
    std::pair<int64_t, std::vector<file_extent>> lba;
    ndp_req->num_finished_jobs = ndp_req->num_jobs = 0;
    ndp_req->total_io_time = 0;

    if (ndp_req->reverse)
    { // TODO
      printf("unsupported reverse operation...\n");
      return -1;
    }
    else if (ndp_req->dir_flag)
    {
      //(ndp_req, input_name, ndp_face_detection_complete);
      switch (ndp_req->task_type)
      {
      case FACE_DETECTION:
      case COMPRESS: //主要是对*bmp文件操作
        //查找所有的bmp, ppm文件
        {
          std::map<std::string, std::pair<int64_t, std::vector<file_extent>>>::iterator it = file_lbas_map.lower_bound(ndp_req->read_path);
          while (it != file_lbas_map.end() && strncmp(it->first.c_str(), ndp_req->read_path, len) == 0)
          {
            //文件前缀和low_bound一样时
            int sublen = it->first.size();
            if (it->second.first > 0) //最后三个字符 "ppm"
            {
              bool add = (ndp_req->task_type == FACE_DETECTION && it->first.substr(sublen - 4) == ".bmp") ||
                         (ndp_req->task_type == COMPRESS && it->first.substr(sublen - 4) == ".ppm");
              if (!add)
              {
                it++;
                continue;
              }

              ndp_req->num_jobs++;
              std::cout << it->first << std::endl;
              //生成子请求并记录
              struct ndp_subrequest *sub_req = (struct ndp_subrequest *)malloc(sizeof(struct ndp_subrequest));
              strcpy(sub_req->read_file, it->first.c_str());
              sub_req->req = ndp_req;
              sub_req->read_ptr = NULL;
              sub_req->sub_id = subID++;
              //红黑树和链表管理。。。
              search_table[ndp_req->id].push_back(sub_req);
            }
            it++;
          }
        }
        break;
      case DECOMPRESS:
        //查找所有的bmp文件
        {
          std::map<std::string, std::pair<int64_t, std::vector<file_extent>>>::iterator it = file_lbas_map.lower_bound(ndp_req->read_path);
          while (it != file_lbas_map.end() && strncmp(it->first.c_str(), ndp_req->read_path, len) == 0)
          {
            int sublen = it->first.size();
            if (it->second.first > 0 && it->first.substr(sublen - 4) == ".dds") //最后三个字符
            {
              ndp_req->num_jobs++;
              //std::cout << it->first << std::endl;
              //生成子请求并记录
              struct ndp_subrequest *sub_req = (struct ndp_subrequest *)malloc(sizeof(struct ndp_subrequest));
              strcpy(sub_req->read_file, it->first.c_str());
              sub_req->req = ndp_req;
              sub_req->read_ptr = NULL;
              sub_req->sub_id = subID++;
              //红黑树和链表管理。。。
              search_table[ndp_req->id].push_back(sub_req);
            }
            it++;
          }
        }
        break;
      default:
        break;
      }
    }
    else
    {
      if (len > 0)
      { //传递了一个目录文件过来
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

    /* 任务正式开始 */
    ndp_req->start_time = spdk_get_ticks();

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
          // printf("direct IO ...\n");
          ret = ndp_compute_face_detection(sub_req);
        }
      }
      break;

      case COMPRESS:
      {
        ret = ndp_read_file(sub_req, sub_req->read_file, ndp_compress_complete, sub_req);

        if (!ret && !ndp_req->spdk_read_flag)
        {
          ret = ndp_compute_compress(sub_req);
        }
      }
      break;
      case DECOMPRESS:
      {
        ret = ndp_read_file(sub_req, sub_req->read_file, ndp_decompress_complete, sub_req);

        if (!ret && !ndp_req->spdk_read_flag)
        {
          ret = ndp_compute_decompress(sub_req);
        }
      }
      break;

      default:
        return -1;
        break;
      }
    }
    if (!ndp_req->spdk_read_flag)
    {
      ndp_req->end_time = spdk_get_ticks(); // ndp_req->start_time= spdk_get_ticks();
      spdk_log_set_flag("ndp");
      SPDK_INFOLOG(ndp, "total jobs %d, total task excution time [%lu], IO time[%lu]\n", ndp_req->num_jobs, (ndp_req->end_time - ndp_req->start_time) / 3500, ndp_req->total_io_time);
      spdk_log_clear_flag("ndp");
      // spdk_nvmf_request_complete(ndp_req->nvmf_req);
      // free(ndp_req);
    }
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
  int offset = 0;

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

    int blk_size = getpagesize();
    int readlen = (lba.first + blk_size - 1) / blk_size * blk_size;
    unsigned char *tempbuf;
    sub_req->malloc_time = spdk_get_ticks();
    posix_memalign((void **)&tempbuf, blk_size, readlen);
    sub_req->read_ptr = tempbuf;

    sub_req->start_io_time = spdk_get_ticks();
    fd = open(input_name, O_DIRECT | O_RDONLY);
    if (fd < 0)
      return -ENOENT;
    cnt = read(fd, tempbuf, readlen);
    close(fd);
    sub_req->end_io_time = spdk_get_ticks();
    ndp_req->total_io_time += (sub_req->end_io_time - sub_req->start_io_time) / 3500;
    assert(cnt > 0);
    // callback_fn(ndp_req->arg);
    //  SPDK_INFOLOG(ndp, "malloc time: %.2f us, diret IO time: %.2f us.\n", 1000000.0 * (ndp_req->start_io_time - ndp_req->malloc_time) / spdk_get_ticks_hz(), 1000000.0 * (spdk_get_ticks() - ndp_req->start_io_time) / spdk_get_ticks_hz());
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
                                        // checkCudaErrors(cudaMalloc((void **)&(gAlloc.decompressTask.inputImage), MAX_COMPRESS_SIZE));
#ifdef ZERO_COPY
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
    checkCudaErrors(cudaHostAlloc((void **)&(gAlloc.compressTask.decompressResult[0]), MAX_COMPRESS_SIZE * 8, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&gAlloc.compressTask.devHostDataToDevice, gAlloc.compressTask.decompressResult[0], 0));
#else

    for (int i = 0; i < MAX_CUDA_PICTURES; i++)
    {
      checkCudaErrors(cudaMalloc((void **)&(gAlloc.compressTask.inputImage[i]), 8 * MAX_COMPRESS_SIZE));
      checkCudaErrors(cudaMalloc((void **)&(gAlloc.compressTask.compressResult[i]), MAX_COMPRESS_SIZE));

      checkCudaErrors(cudaMalloc((void **)&(gAlloc.decompressTask.inputImage[i]), MAX_COMPRESS_SIZE));
      checkCudaErrors(cudaMalloc((void **)&(gAlloc.decompressTask.decompressResult[i]), 8 * MAX_COMPRESS_SIZE));
    }
    checkCudaErrors(cudaMalloc((void **)&(gAlloc.compressTask.d_permutations), 1024 * sizeof(uint)));
    gAlloc.compressTask.permutations = (uint *)malloc(1024 * sizeof(uint));
    // compute permutation
    computePermutations(gAlloc.compressTask.permutations);
    checkCudaErrors(cudaMemcpy(gAlloc.compressTask.d_permutations, gAlloc.compressTask.permutations, 1024 * sizeof(uint),
                               cudaMemcpyHostToDevice));
#endif
    spdk_log_clear_flag("ndp");
    return 0;
  }

  int ndp_free(void)
  {
    for (int i = 0; i < MAX_CUDA_PICTURES; i++)
    {
      checkCudaErrors(cudaFree((void *)gAlloc.compressTask.inputImage[i]));
      checkCudaErrors(cudaFree((void *)gAlloc.compressTask.compressResult[i]));
      checkCudaErrors(cudaFree((void *)gAlloc.decompressTask.inputImage[i]));
      checkCudaErrors(cudaFree((void *)gAlloc.decompressTask.decompressResult[i]));
    }
    cudaFree((void *)(gAlloc.compressTask.d_permutations));
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
  int ndp_decompress(struct ndp_subrequest *sub_req, const char *filename, uint8_t **result) //选不选择NDP加速
  {
    // Gflops/s
    //统计NDP计算时间
    uint64_t ticks_per_second, start_ticks, end_ticks;
    spdk_log_clear_flag("ndp");
    ticks_per_second = spdk_get_ticks_hz();
    start_ticks = spdk_get_ticks();
    int opt = sub_req->req->accel;

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
    BlockDXT1 *device_ptr = (BlockDXT1 *)gAlloc.decompressTask.inputImage[0];
    Color32 *device_result_ptr = (Color32 *)gAlloc.decompressTask.decompressResult[0];
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
    //普通解压缩操作
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
        cudaMemcpy(device_ptr, compressImageBuffer, header.pitch, cudaMemcpyHostToDevice);
        cuda_memcpy_ticks = spdk_get_ticks();
        if ((1000 * (cuda_memcpy_ticks - temp_start_ticks) / ticks_per_second) > 1)
        {
          SPDK_INFOLOG(ndp, "[%d] NDP's memcpy time too high... %ld (us)\n", k, 1000000 * (cuda_memcpy_ticks - temp_start_ticks) / ticks_per_second);
        }
        ndp_accelerate_decompress<<<numBlocks, threadPerBlock>>>(device_ptr, device_result_ptr, H, W);
        checkCudaErrors(cudaDeviceSynchronize());

        temp_end_ticks = spdk_get_ticks();
        cudaMemcpy(h_result, device_result_ptr, w * h * 4, cudaMemcpyDeviceToHost);
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
    return 0;
  }
};

/* 读取存储相关的数据后, 执行NDP加速 */

static void ndp_wordcount_complete(struct spdk_bdev_io *bdev_io, bool success,
                                   void *cb_arg)
{
  struct ndp_subrequest *sub_req = (struct ndp_subrequest *)cb_arg;
  uint64_t ticks_per_second;
  // struct spdk_nvme_cpl *response = &sub_req->req->nvmf_req->rsp->nvme_cpl;
  int sc = 0, sct = 0;
  uint32_t cdw0 = 0;

  // cnn_flag = ndp_req->accel;                               //是否使用加速场景, cdw13
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

#define ERROR_THRESHOLD 0.02f

#define NUM_THREADS 64 // Number of threads per block.
#include <cooperative_groups.h>
#include <float.h>
#include <vector_types.h>
#include "ndp/helper_timer.h"
namespace cg = cooperative_groups;

template <class T>
__device__ inline void swap(T &a, T &b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

inline bool __loadPPM_from_mem(const char *file_data, unsigned char **data, unsigned int *w,
                               unsigned int *h, unsigned int *channels)
{

  int offset = 0;
  // check header
  char header[helper_image_internal::PGMHeaderSize];
  memcpy(header, file_data + offset, 3);
  offset += 3;
  if (strncmp(header, "P5", 2) == 0)
  {
    *channels = 1;
  }
  else if (strncmp(header, "P6", 2) == 0)
  {
    *channels = 3;
  }
  else
  {
    std::cerr << "__LoadPPM() : File is not a PPM or PGM image" << std::endl;
    *channels = 0;
    return false;
  }

  // parse header, read maxval, width and height
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int maxval = 0;
  unsigned int i = 0;
  int n;
  while (i < 3)
  {
    int k = 0;
    while (!isspace(*(file_data + offset)))
    {
      header[k++] = *(file_data + offset);
      ++offset;
    }
    ++offset;
    header[k] = '\0';
    if (header[0] == '#')
    {
      continue;
    }

    if (i == 0)
    {
      i += SSCANF(header, "%u %u %u", &width, &height, &maxval);
    }
    else if (i == 1)
    {
      i += SSCANF(header, "%u %u", &height, &maxval);
    }
    else if (i == 2)
    {
      i += SSCANF(header, "%u", &maxval);
    }
  }

  // check if given handle for the data is initialized
  if (NULL != *data)
  {
    if (*w != width || *h != height)
    {
      std::cerr << "__LoadPPM() : Invalid image dimensions." << std::endl;
    }
  }
  else
  {
    --offset;
    *data = (unsigned char *)(file_data + offset);
    *w = width;
    *h = height;
  }
  // // read and close file
  //  fread(*data, sizeof(unsigned char), width * height * *channels, fp) ==
  //     0)
  // fclose(fp);

  return true;
}

inline bool sdkLoadPPM4ub_from_mem(const char *file_data, unsigned char **data,
                                   unsigned int *w, unsigned int *h)
{
  unsigned char *idata = 0;
  unsigned int channels;

  if (__loadPPM_from_mem(file_data, &idata, w, h, &channels))
  {
    // pad 4th component
    int size = *w * *h;
    // keep the original pointer
    unsigned char *idata_orig = idata;
    *data = (unsigned char *)malloc(sizeof(unsigned char) * size * 4);
    unsigned char *ptr = *data;

    for (int i = 0; i < size; i++)
    {
      *ptr++ = *idata++;
      *ptr++ = *idata++;
      *ptr++ = *idata++;
      *ptr++ = 0;
    }

    // free(idata_orig);
    return true;
  }
  else
  {
    // free(idata);
    return false;
  }
}

__global__ void compress(const uint *permutations, const uint *image,
                         uint2 *result, int blockOffset);

static int ndp_compute_compress(struct ndp_subrequest *sub_req)
{
  struct ndp_request *ndp_req = sub_req->req;
  bool accel = sub_req->req->accel;
  int sc = 0, sct = 0;
  int ret = 0;
  uint32_t cdw0 = 0;
  unsigned char *src = NULL; //需要free掉
  uint *block_image = NULL;

  uint W, H;
  /* 解析文件 */
  if (!sdkLoadPPM4ub_from_mem((const char *)(sub_req->read_ptr), &src, &W, &H))
  {
    printf("Error, unable to open source image file <%s>\n", sub_req->read_file);
    return -EXIT_FAILURE;
  }

  uint w = W, h = H;
  //计算结果文件的大小
  const uint compressedSize = (w / 4) * (h / 4) * 8;
  uint *h_result = (uint *)malloc(compressedSize);

  /* 执行压缩文件 */
  if (!accel)
  {
    uint64_t temp_start_compute_time = spdk_get_ticks();
    CompressImageDXT1((const BYTE *)src, (BYTE *)h_result, (int)W, (int)H);
     SPDK_INFOLOG(ndp, "CompressImageDXT1 time:[%lu]\n", (spdk_get_ticks() - temp_start_compute_time) / 3500);
  }
  else
  {
    const uint memSize = w * h * 4;
    block_image = (uint *)malloc(memSize);
    // Convert linear image to block linear.
    for (uint by = 0; by < h / 4; by++)
    {
      for (uint bx = 0; bx < w / 4; bx++)
      {
        for (int i = 0; i < 16; i++)
        {
          const int x = i & 3;
          const int y = i / 4;
          block_image[(by * w / 4 + bx) * 16 + i] =
              ((uint *)src)[(by * 4 + y) * 4 * (W / 4) + bx * 4 + x];
        }
      }
    }

    uint blocks = ((w + 3) / 4) *
                  ((h + 3) / 4); // rounds up by 1 block in each dim if %4 != 0

#ifdef CHECK_GPU
    int devID;
    cudaDeviceProp deviceProp;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
#endif
    uint *d_data = (uint *)gAlloc.compressTask.inputImage[sub_req->sub_id];
    uint *d_result = (uint *)gAlloc.compressTask.compressResult[sub_req->sub_id];

    checkCudaErrors(
        cudaMemcpy(d_data, block_image, memSize, cudaMemcpyHostToDevice)); //加载bmp文件, 需要预处理不可。
    // Restrict the numbers of blocks to launch on low end GPUs to avoid kernel
    // timeout
    int blocksPerLaunch = blocks; // min(blocks, 768 * deviceProp.multiProcessorCount);

    printf("Running DXT Compression on %u x %u image...\n", w, h);
    printf("\n%u Blocks, %u Threads per Block, %u Threads in Grid...\n\n", blocks,
           NUM_THREADS, blocks * NUM_THREADS);
    uint *d_permutations = (uint *)gAlloc.compressTask.d_permutations;

    //压缩数据块
    for (int j = 0; j < (int)blocks; j += blocksPerLaunch)
    {
      compress<<<min(blocksPerLaunch, blocks - j), NUM_THREADS>>>(
          d_permutations, d_data, (uint2 *)d_result, j);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(
        cudaMemcpy(h_result, d_result, compressedSize, cudaMemcpyDeviceToHost));
  }
  sub_req->end_compute_time = spdk_get_ticks(); //计算结束时间
  //打印时间戳
  //SPDK_INFOLOG(ndp, "file[%s], exclude malloc, total time: %lu, SPDK malloc time: %lu , SPDK IO consume: %lu, computation time: %lu\n", sub_req->read_file, (sub_req->end_compute_time - sub_req->start_io_time) / 3500, (sub_req->start_io_time - sub_req->malloc_time) / 3500, (sub_req->end_io_time - sub_req->start_io_time) / 3500, (sub_req->end_compute_time - sub_req->end_io_time) / 3500);

  // Write out result data to DDS file
  char output_filename[1024];
  strcpy(output_filename, sub_req->read_file);
  strcpy(output_filename + strlen(sub_req->read_file) - 4, "_spdk.dxt_compress");
  FILE *fp = fopen(output_filename, "wb");

  if (fp == 0)
  {
    printf("Error, unable to open output image <%s>\n", output_filename);
    exit(EXIT_FAILURE);
  }
  DDSHeader header;
  header.fourcc = FOURCC_DDS;
  header.size = 124;
  header.flags = (DDSD_WIDTH | DDSD_HEIGHT | DDSD_CAPS | DDSD_PIXELFORMAT |
                  DDSD_LINEARSIZE);
  header.height = h;
  header.width = w;
  header.pitch = compressedSize;
  header.depth = 0;
  header.mipmapcount = 0;
  memset(header.reserved, 0, sizeof(header.reserved));
  header.pf.size = 32;
  header.pf.flags = DDPF_FOURCC;
  header.pf.fourcc = FOURCC_DXT1;
  header.pf.bitcount = 0;
  header.pf.rmask = 0;
  header.pf.gmask = 0;
  header.pf.bmask = 0;
  header.pf.amask = 0;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  header.caps.caps2 = 0;
  header.caps.caps3 = 0;
  header.caps.caps4 = 0;
  header.notused = 0;
  fwrite(&header, sizeof(DDSHeader), 1, fp);
  fwrite(h_result, compressedSize, 1, fp);
  fclose(fp);
  free(h_result);
  free(src);
  if (accel)
    free(block_image);
  return 0;
}

//__constant__ float3 kColorMetric = { 0.2126f, 0.7152f, 0.0722f };
__constant__ float3 kColorMetric = {1.0f, 1.0f, 1.0f};

inline __device__ __host__ float3 firstEigenVector(float matrix[6])
{
  // 8 iterations seems to be more than enough.

  float3 v = make_float3(1.0f, 1.0f, 1.0f);

  for (int i = 0; i < 8; i++)
  {
    float x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
    float y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
    float z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
    float m = max(max(x, y), z);
    float iv = 1.0f / m;
    v = make_float3(x * iv, y * iv, z * iv);
  }

  return v;
}

inline __device__ void colorSums(const float3 *colors, float3 *sums,
                                 cg::thread_group tile)
{
  const int idx = threadIdx.x;

  sums[idx] = colors[idx];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 8];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 4];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 2];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 1];
}

inline __device__ float3 bestFitLine(const float3 *colors, float3 color_sum,
                                     cg::thread_group tile)
{
  // Compute covariance matrix of the given colors.
  const int idx = threadIdx.x;

  float3 diff = colors[idx] - color_sum * (1.0f / 16.0f);

  // @@ Eliminate two-way bank conflicts here.
  // @@ It seems that doing that and unrolling the reduction doesn't help...
  __shared__ float covariance[16 * 6];

  covariance[6 * idx + 0] = diff.x * diff.x; // 0, 6, 12, 2, 8, 14, 4, 10, 0
  covariance[6 * idx + 1] = diff.x * diff.y;
  covariance[6 * idx + 2] = diff.x * diff.z;
  covariance[6 * idx + 3] = diff.y * diff.y;
  covariance[6 * idx + 4] = diff.y * diff.z;
  covariance[6 * idx + 5] = diff.z * diff.z;

  cg::sync(tile);
  for (int d = 8; d > 0; d >>= 1)
  {
    if (idx < d)
    {
      covariance[6 * idx + 0] += covariance[6 * (idx + d) + 0];
      covariance[6 * idx + 1] += covariance[6 * (idx + d) + 1];
      covariance[6 * idx + 2] += covariance[6 * (idx + d) + 2];
      covariance[6 * idx + 3] += covariance[6 * (idx + d) + 3];
      covariance[6 * idx + 4] += covariance[6 * (idx + d) + 4];
      covariance[6 * idx + 5] += covariance[6 * (idx + d) + 5];
    }
    cg::sync(tile);
  }

  // Compute first eigen vector.
  return firstEigenVector(covariance);
}

////////////////////////////////////////////////////////////////////////////////
// Sort colors
////////////////////////////////////////////////////////////////////////////////
__device__ void sortColors(const float *values, int *ranks,
                           cg::thread_group tile)
{
  const int tid = threadIdx.x;

  int rank = 0;

#pragma unroll

  for (int i = 0; i < 16; i++)
  {
    rank += (values[i] < values[tid]);
  }

  ranks[tid] = rank;

  cg::sync(tile);

  // Resolve elements with the same index.
  for (int i = 0; i < 15; i++)
  {
    if (tid > i && ranks[tid] == ranks[i])
    {
      ++ranks[tid];
    }
    cg::sync(tile);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Load color block to shared mem
////////////////////////////////////////////////////////////////////////////////
__device__ void loadColorBlock(const uint *image, float3 colors[16],
                               float3 sums[16], int xrefs[16], int blockOffset,
                               cg::thread_block cta)
{
  const int bid = blockIdx.x + blockOffset;
  const int idx = threadIdx.x;

  __shared__ float dps[16];

  float3 tmp;

  cg::thread_group tile = cg::tiled_partition(cta, 16);

  if (idx < 16)
  {
    // Read color and copy to shared mem.
    uint c = image[(bid)*16 + idx];

    colors[idx].x = ((c >> 0) & 0xFF) * (1.0f / 255.0f);
    colors[idx].y = ((c >> 8) & 0xFF) * (1.0f / 255.0f);
    colors[idx].z = ((c >> 16) & 0xFF) * (1.0f / 255.0f);

    cg::sync(tile);
    // Sort colors along the best fit line.
    colorSums(colors, sums, tile);

    cg::sync(tile);

    float3 axis = bestFitLine(colors, sums[0], tile); //这是？

    cg::sync(tile);

    dps[idx] = dot(colors[idx], axis);

    cg::sync(tile);

    sortColors(dps, xrefs, tile);

    cg::sync(tile);

    tmp = colors[idx];

    cg::sync(tile);

    colors[xrefs[idx]] = tmp;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Round color to RGB565 and expand
////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 roundAndExpand(float3 v, ushort *w)
{
  v.x = rintf(__saturatef(v.x) * 31.0f);
  v.y = rintf(__saturatef(v.y) * 63.0f);
  v.z = rintf(__saturatef(v.z) * 31.0f);

  *w = ((ushort)v.x << 11) | ((ushort)v.y << 5) | (ushort)v.z;
  v.x *= 0.03227752766457f; // approximate integer bit expansion.
  v.y *= 0.01583151765563f;
  v.z *= 0.03227752766457f;
  return v;
}

__constant__ float alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
__constant__ float alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
__constant__ const int prods4[4] = {0x090000, 0x000900, 0x040102, 0x010402};
__constant__ const int prods3[4] = {0x040000, 0x000400, 0x040101, 0x010401};

#define USE_TABLES 1

////////////////////////////////////////////////////////////////////////////////
// Evaluate permutations
////////////////////////////////////////////////////////////////////////////////
static __device__ float evalPermutation4(const float3 *colors, uint permutation,
                                         ushort *start, ushort *end,
                                         float3 color_sum)
{
// Compute endpoints using least squares.
#if USE_TABLES
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  int akku = 0;

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++)
  {
    const uint bits = permutation >> (2 * i);

    alphax_sum += alphaTable4[bits & 3] * colors[i];
    akku += prods4[bits & 3];
  }

  float alpha2_sum = float(akku >> 16);
  float beta2_sum = float((akku >> 8) & 0xff);
  float alphabeta_sum = float((akku >> 0) & 0xff);
  float3 betax_sum = (9.0f * color_sum) - alphax_sum;
#else
  float alpha2_sum = 0.0f;
  float beta2_sum = 0.0f;
  float alphabeta_sum = 0.0f;
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++)
  {
    const uint bits = permutation >> (2 * i);

    float beta = (bits & 1);

    if (bits & 2)
    {
      beta = (1 + beta) * (1.0f / 3.0f);
    }

    float alpha = 1.0f - beta;

    alpha2_sum += alpha * alpha;
    beta2_sum += beta * beta;
    alphabeta_sum += alpha * beta;
    alphax_sum += alpha * colors[i];
  }

  float3 betax_sum = color_sum - alphax_sum;
#endif

  // alpha2, beta2, alphabeta and factor could be precomputed for each
  // permutation, but it's faster to recompute them.
  const float factor =
      1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

  float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
  float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

  // Round a, b to the closest 5-6-5 color and expand...
  a = roundAndExpand(a, start);
  b = roundAndExpand(b, end);

  // compute the error
  float3 e = a * a * alpha2_sum + b * b * beta2_sum +
             2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

  return (0.111111111111f) * dot(e, kColorMetric);
}

static __device__ float evalPermutation3(const float3 *colors, uint permutation,
                                         ushort *start, ushort *end,
                                         float3 color_sum)
{
// Compute endpoints using least squares.
#if USE_TABLES
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  int akku = 0;

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++)
  {
    const uint bits = permutation >> (2 * i);

    alphax_sum += alphaTable3[bits & 3] * colors[i];
    akku += prods3[bits & 3];
  }

  float alpha2_sum = float(akku >> 16);
  float beta2_sum = float((akku >> 8) & 0xff);
  float alphabeta_sum = float((akku >> 0) & 0xff);
  float3 betax_sum = (4.0f * color_sum) - alphax_sum;
#else
  float alpha2_sum = 0.0f;
  float beta2_sum = 0.0f;
  float alphabeta_sum = 0.0f;
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++)
  {
    const uint bits = permutation >> (2 * i);

    float beta = (bits & 1);

    if (bits & 2)
    {
      beta = 0.5f;
    }

    float alpha = 1.0f - beta;

    alpha2_sum += alpha * alpha;
    beta2_sum += beta * beta;
    alphabeta_sum += alpha * beta;
    alphax_sum += alpha * colors[i];
  }

  float3 betax_sum = color_sum - alphax_sum;
#endif

  const float factor =
      1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

  float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
  float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

  // Round a, b to the closest 5-6-5 color and expand...
  a = roundAndExpand(a, start);
  b = roundAndExpand(b, end);

  // compute the error
  float3 e = a * a * alpha2_sum + b * b * beta2_sum +
             2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

  return (0.25f) * dot(e, kColorMetric);
}

__device__ void evalAllPermutations(const float3 *colors,
                                    const uint *permutations, ushort &bestStart,
                                    ushort &bestEnd, uint &bestPermutation,
                                    float *errors, float3 color_sum,
                                    cg::thread_block cta)
{
  const int idx = threadIdx.x;

  float bestError = FLT_MAX;

  __shared__ uint s_permutations[160];

  for (int i = 0; i < 16; i++)
  {
    int pidx = idx + NUM_THREADS * i;

    if (pidx >= 992)
    { //为什么是992？
      break;
    }

    ushort start, end;
    uint permutation = permutations[pidx];

    if (pidx < 160)
    {
      s_permutations[pidx] = permutation;
    }

    float error =
        evalPermutation4(colors, permutation, &start, &end, color_sum);

    if (error < bestError)
    {
      bestError = error;
      bestPermutation = permutation;
      bestStart = start;
      bestEnd = end;
    }
  }

  if (bestStart < bestEnd)
  {
    swap(bestEnd, bestStart);
    bestPermutation ^= 0x55555555; // Flip indices.
  }

  cg::sync(cta); // Sync here to ensure s_permutations is valid going forward

  for (int i = 0; i < 3; i++)
  {
    int pidx = idx + NUM_THREADS * i;

    if (pidx >= 160)
    {
      break;
    }

    ushort start, end;
    uint permutation = s_permutations[pidx];
    float error =
        evalPermutation3(colors, permutation, &start, &end, color_sum);

    if (error < bestError)
    {
      bestError = error;
      bestPermutation = permutation;
      bestStart = start;
      bestEnd = end;

      if (bestStart > bestEnd)
      {
        swap(bestEnd, bestStart);
        bestPermutation ^=
            (~bestPermutation >> 1) & 0x55555555; // Flip indices.
      }
    }
  }

  errors[idx] = bestError;
}

////////////////////////////////////////////////////////////////////////////////
// Find index with minimum error
////////////////////////////////////////////////////////////////////////////////
__device__ int findMinError(float *errors, cg::thread_block cta)
{
  const int idx = threadIdx.x;
  __shared__ int indices[NUM_THREADS];
  indices[idx] = idx;

  cg::sync(cta);

  for (int d = NUM_THREADS / 2; d > 0; d >>= 1)
  {
    float err0 = errors[idx];
    float err1 = (idx + d) < NUM_THREADS ? errors[idx + d] : FLT_MAX;
    int index1 = (idx + d) < NUM_THREADS ? indices[idx + d] : 0;

    cg::sync(cta);

    if (err1 < err0)
    {
      errors[idx] = err1;
      indices[idx] = index1;
    }

    cg::sync(cta);
  }

  return indices[0];
}

////////////////////////////////////////////////////////////////////////////////
// Save DXT block
////////////////////////////////////////////////////////////////////////////////
__device__ void saveBlockDXT1(ushort start, ushort end, uint permutation,
                              int xrefs[16], uint2 *result, int blockOffset)
{
  const int bid = blockIdx.x + blockOffset;

  if (start == end)
  {
    permutation = 0;
  }

  // Reorder permutation.
  uint indices = 0;

  for (int i = 0; i < 16; i++)
  {
    int ref = xrefs[i];
    indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
  }

  // Write endpoints.
  result[bid].x = (end << 16) | start;

  // Write palette indices.
  result[bid].y = indices;
}

__global__ void compress(const uint *permutations, const uint *image,
                         uint2 *result, int blockOffset)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  const int idx = threadIdx.x;

  __shared__ float3 colors[16];
  __shared__ float3 sums[16];
  __shared__ int xrefs[16];

  loadColorBlock(image, colors, sums, xrefs, blockOffset, cta);

  cg::sync(cta);

  ushort bestStart, bestEnd;
  uint bestPermutation;

  __shared__ float errors[NUM_THREADS];

  evalAllPermutations(colors, permutations, bestStart, bestEnd, bestPermutation,
                      errors, sums[0], cta);

  // Use a parallel reduction to find minimum error.
  const int minIdx = findMinError(errors, cta); //寻找损失函数最小的一个点

  cg::sync(cta);

  // Only write the result of the winner thread.
  if (idx == minIdx)
  {
    saveBlockDXT1(bestStart, bestEnd, bestPermutation, xrefs, result,
                  blockOffset);
  }
}

static void computePermutations(uint *permutations)
{
  int indices[16];
  int num = 0;

  // 3 element permutations:

  // first cluster [0,i) is at the start
  for (int m = 0; m < 16; ++m)
  {
    indices[m] = 0;
  }

  const int imax = 15;

  for (int i = imax; i >= 0; --i)
  {
    // second cluster [i,j) is half along
    for (int m = i; m < 16; ++m)
    {
      indices[m] = 2;
    }

    const int jmax = (i == 0) ? 15 : 16;

    for (int j = jmax; j >= i; --j)
    {
      // last cluster [j,k) is at the end
      if (j < 16)
      {
        indices[j] = 1;
      }

      uint permutation = 0;

      for (int p = 0; p < 16; p++)
      {
        permutation |= indices[p] << (p * 2);
        // permutation |= indices[15-p] << (p * 2);
      }

      permutations[num] = permutation;

      num++;
    }
  }

  assert(num == 151);

  for (int i = 0; i < 9; i++)
  {
    permutations[num] = 0x000AA555;
    num++;
  }

  assert(num == 160);

  // Append 4 element permutations:

  // first cluster [0,i) is at the start
  for (int m = 0; m < 16; ++m)
  {
    indices[m] = 0;
  }

  for (int i = imax; i >= 0; --i)
  {
    // second cluster [i,j) is one third along
    for (int m = i; m < 16; ++m)
    {
      indices[m] = 2;
    }

    const int jmax = (i == 0) ? 15 : 16;

    for (int j = jmax; j >= i; --j)
    {
      // third cluster [j,k) is two thirds along
      for (int m = j; m < 16; ++m)
      {
        indices[m] = 3;
      }

      int kmax = (j == 0) ? 15 : 16;

      for (int k = kmax; k >= j; --k)
      {
        // last cluster [k,n) is at the end
        if (k < 16)
        {
          indices[k] = 1;
        }

        uint permutation = 0;

        bool hasThree = false;

        for (int p = 0; p < 16; p++)
        {
          permutation |= indices[p] << (p * 2);
          // permutation |= indices[15-p] << (p * 2);

          if (indices[p] == 3)
            hasThree = true;
        }

        if (hasThree)
        {
          permutations[num] = permutation;
          num++;
        }
      }
    }
  }

  assert(num == 975);

  // 1024 - 969 - 7 = 48 extra elements

  // It would be nice to set these extra elements with better values...
  for (int i = 0; i < 49; i++)
  {
    permutations[num] = 0x00AAFF55;
    num++;
  }

  assert(num == 1024);
}