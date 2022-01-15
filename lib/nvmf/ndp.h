
#ifndef __NDP_H__
#define __NDP_H__

#define DEFAULT_NDP_DIR "/mnt/ndp"
#define BDEV_NAME "Malloc0" 
#define BLK_SIZE 512
#define BLK_SHIFT_BIT 9
#define MB (1024 * 1024)
#define MAX_GRAPH (1024 * 1024)
#define MAX_COMPRESS_SIZE (MAX_GRAPH) //分大一点似乎好一点？
struct ndp_request
{
    struct spdk_nvmf_request *nvmf_req; //找到nvme命令by cmd->nvme_cmd;
    struct spdk_bdev_desc *desc;
    struct spdk_io_channel *io_ch;
    uint64_t start_io_time, end_io_time;
    int read_bdev_blocks;
    int total_bdev_blocks;
    int total_len;

    union 
    {
        char *read_ptr;
    } ptr;
    
};

#endif
