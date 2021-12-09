
#ifndef __NDP_H__
#define __NDP_H__

//#include "spdk/bdev.h"
//#include "spdk/nvmf_transport.h"

struct ndp_request
{
    struct spdk_nvmf_request *nvmf_req;
    int read_bdev_blocks;
    int total_bdev_blocks;
};


#endif
