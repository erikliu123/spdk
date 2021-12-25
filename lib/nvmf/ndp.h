
#ifndef __NDP_H__
#define __NDP_H__

#define DEFAULT_NDP_DIR "/mnt/ndp/"
#define MB (1024 * 1024)
#define MAX_GRAPH (1024 * 1024)
#define MAX_COMPRESS_SIZE (MAX_GRAPH) //分大一点似乎好一点？
struct ndp_request
{
    struct spdk_nvmf_request *nvmf_req;
    int read_bdev_blocks;
    int total_bdev_blocks;
};


#endif
