
#ifndef __NDP_H__
#define __NDP_H__

#define CONFIG_NDP

#define DEFAULT_NDP_DIR "/mnt/ndp"
#define BDEV_NAME "Malloc0" 
#define BLK_SIZE 512
#define BLK_SHIFT_BIT 9
#define FILE_SYS_BIT 12
#define MB (1024 * 1024)
#define MAX_GRAPH (1024 * 1024)
#define MAX_COMPRESS_SIZE (MAX_GRAPH) //分大一点似乎好一点？

enum NDP_TASK {
    FACE_DETECTION=0, 
    COMPRESS, 
    DECOMPRESS,
    MAX_NDP_TASK
};

enum FACE_DETECTION_FLAG
{
    SPDK_READ_FALSG = 1,
    FEATURE_FLAG = 2,
};

enum  NDP_ERROR_TYPE
{
    ERR_NO_FACE = 140,
};


#pragma pack(4)
//map <ID, ndp_request *>
//subrequest包含ndq_request, id, 文件名， 读取的read_bdev_blocks， 总共的total_bdev_blocks
struct ndp_request
{
#ifdef CONFIG_NDP
    struct spdk_nvmf_request *nvmf_json_req;
    struct spdk_nvmf_request *nvmf_req; //找到nvme命令by cmd->nvme_cmd;
    struct spdk_bdev_desc *desc;
    struct spdk_io_channel *io_ch;
    int read_bdev_blocks;
    int total_bdev_blocks;
    //有多少个文件任务
    int num_jobs;
    int num_finished_jobs;
    //std::vector<std::string> read_file_vec, read_dir;
#endif    
    int id;//0x81，下发时候设置的ID。
    int task_type;
    bool dir_flag;//是在某目录夹操作还是对某个文件进行指定的操作。
    bool reverse;//是否递归查找
    char read_path[256];//文件/目录路径？
    //uint64_t malloc_time, start_io_time, end_io_time, end_compute_time;
    uint64_t start_time, end_time;
    int total_len;    
    bool accel, spdk_read_flag;
    int timeout_ms;

    union {
        struct //人脸检测任务
        {
            bool cnn_flag;//=accel
            bool face_feature_flag;
        } face_detection;

        struct{//文件查找，在某目录下查找search_name相关的
            char *name;//查找的文件
        }search_file;

        struct{//单词查找，在某文件或者目录夹下执行
            char *word;//查找的文件
        }search_word;

    } task;   
}; //__attribute__((packed));

struct ndp_subrequest
{
    struct ndp_request *req;
    int sub_id;
    char read_file[256];
    int read_bdev_blocks;
    int total_bdev_blocks;  
    int total_len;
    
    unsigned char *read_ptr;//读取的文件数据，
    uint64_t malloc_time, start_io_time, end_io_time, end_compute_time;  
    bool finished;
};

// int register_ndp_task(struct ndp_request *ndp_req);
// struct ndp_request * get_ndp_req_from_id(int id);

// int alloc_and_start_sub_req(struct ndp_request *ndp_req);
void spdk_ndp_request_complete(struct ndp_subrequest *sub_req, int status);
int ndp_read_file(struct ndp_subrequest *sub_req, char *input_name, void (*callback_fn)(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg), void* arg);


#pragma pack()
#endif
