#include <iostream>
#include <stdio.h>
#include <map>
#include <unordered_map>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>

#include "md5.h"
#include "md5kernel.h"

using namespace std;

uint64_t start_lba = 57 * 512, lba_count = 1024;
int times = 50;
char zero_buf[4096];
bool g_accel = true;

void help(char *app)
{
    printf("%s: /dev/nvme 1(accel) 10(loop times)\n", app);
    return;
}
struct md5_key
{
    uint8_t md5[16];

    bool operator==(const struct md5_key &b) const
    {

        for (int i = 0; i < 16; i++)
        {
            if (md5[i] != b.md5[i])
                return false;
        }
        return true;
    }
};

uint64_t hash_int(uint64_t num)
{
}
struct hashFunc
{
    size_t operator()(const md5_key &in) const
    {
        // how to hash 128-bit md5 to 64 bit
        // simply choose the last two
        const uint8_t *md5 = in.md5;
        auto fn = hash<int>();
        size_t res = 0;
        for (int i = 0; i < 16; i++)
        {
            res = (uint64_t)fn(md5[i] << 4) ^ (uint64_t)fn(res);
        }
        // cout<<res<<endl;
        return res;
    }
};
md5_key md5_zero, md5_digest_arr[1024];
unordered_map<md5_key, uint64_t, hashFunc> md5_table;

void printf_md5(uint8_t *digest)
{
    printf("md5: ");
    for (int i = 0; i < 8; i++)
    {
        printf("%x ", digest[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{

    int deplicate_blocks = 0;
    int lba_size = 4096; // 4KB * 1024 * 8 =  32 mb

    //扫描盘的区域
    int fd;
    double iotime = 0.0;
    struct timeval tpstart, iostart, ioend, tpend;
    double timeuse, total_iotime = 0.0, total_timeuse = 0.0;
   
    if (argc < 4)
    {
        help(argv[0]);
        return 0;
    }
    fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0)
        return 0;
    g_accel = (atoi(argv[2]) != 0);
    int s_times = atoi(argv[3]);
    char *read_ptr = NULL; //= (char *)malloc(lba_count * lba_size);
    md5((uint8_t *)(zero_buf), lba_size, md5_zero.md5);
    printf_md5(md5_zero.md5);

    lseek(fd, start_lba * lba_size, SEEK_SET);
    posix_memalign((void **)&read_ptr, lba_size, lba_size * lba_count);
    for (int loop = 0; loop < s_times; loop++)
    {
        preallloc(lba_count * lba_size);
        md5_table.clear();
        gettimeofday(&tpstart, NULL);
        deplicate_blocks = 0;
        start_lba = 57 * 512;
        
        for (int cnt = 0; cnt < times; cnt++)
        {
            gettimeofday(&iostart, NULL);
            read(fd, read_ptr, lba_count * lba_size);
            gettimeofday(&ioend, NULL);
            // fetch data, deduplication may lead  latency becomes higher if don't eliminates duplicate read
            // send nvme request and then get ythe block， maybe exist two
            md5_key result;
            if (g_accel)
            {
                uint64_t consume_time;
                md5WithCuda((uint8_t *)read_ptr, lba_count * lba_size, (uint8_t *)md5_digest_arr, consume_time);
                // printf("consume time: %lu\n", consume_time);
                for (uint64_t i = 0; i < lba_count; i++)
                {
                    // printf_md5(result.md5);
                    if (md5_digest_arr[i] == md5_zero) //|| md5_digest_arr[i].md5[0]==0)
                        continue;
                    if (md5_table.count(md5_digest_arr[i]))
                    { // TODO: caculate time
                        uint64_t lba = md5_table[md5_digest_arr[i]];
                        if (lba == 0)
                            continue;
                        // send duplicate request
                        // printf_md5(md5_digest_arr[i].md5);
                        ++deplicate_blocks;
                        // printf("DEDUP(%lu,%lu) ERROR!!!!!!!!!\n", start_lba+i, lba);
                    }
                    // md5_table
                    else
                    {
                        if (md5_table.size() > 50000)
                            continue;
                        md5_table[md5_digest_arr[i]] = start_lba + i;
                    }
                }
            }

            else
            {
                for (uint64_t i = 0; i < lba_count; i++)
                {
                    //
                    md5((uint8_t *)(read_ptr + i * lba_size), lba_size, result.md5); // accelerate
                    // printf_md5(result.md5);
                    if (result == md5_zero)
                        continue;
                    if (md5_table.count(result))
                    { // TODO: caculate time
                        uint64_t lba = md5_table[result];
                        if (lba == 0)
                            continue;
                        // send duplicate request
                        // printf_md5(result.md5);
                        ++deplicate_blocks;
                        // printf("DEDUP(%lu,%lu) ERROR!!!!!!!!!\n", start_lba+i, lba);
                    }
                    // md5_table
                    else
                    {
                        if (md5_table.size() > 50000)
                            continue;
                        md5_table[result] = start_lba + i;
                    }
                }
            }
            start_lba += lba_count;
            gettimeofday(&tpend, NULL);
            iotime += 1000000 * (ioend.tv_sec - iostart.tv_sec) + ioend.tv_usec - iostart.tv_usec;
            
        }
        timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
        total_timeuse += timeuse;
        total_iotime += iotime;
        sleep(1);
        free_cuda();
    }
    printf("md5 table size:%lu, deuplicates: %d, duplicate rate: %.2f, IO: %.2f, TOTAL: %.2f\n", md5_table.size(), deplicate_blocks, 1.0 * deplicate_blocks / (times * lba_count), iotime/s_times, total_timeuse/s_times);
    return 0;
}
