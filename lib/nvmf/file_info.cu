#define _FILE_OFFSET_BITS 64
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <vector>
#include <map>
#include <vector>
#include "file_info.h"

std::map<std::string, std::pair<int64_t, std::vector<file_extent>>> file_lbas_map;//文件名和对应的文件大小、lba数组
static const unsigned int sector_bytes = 512; // FIXME someday


int walk_fibmap (int fd, struct stat *st, unsigned int blksize, unsigned int sectors_per_block, __u64 start_lba, std::vector<file_extent> &result)//struct file_extent *result)
{
	struct file_extent ext;
	__u64 num_blocks;
	__u64 blk_idx, hole = ~0ULL;

	/*
	 * How many calls to FIBMAP do we need?
	 * FIBMAP returns a filesystem block number (counted from the start of the device)
	 * for each file block.  This can be converted to a disk LBA using the filesystem
	 * blocksize and LBA offset obtained earlier.
	 */
	num_blocks = (st->st_size + blksize - 1) / blksize;
	memset(&ext, 0, sizeof(ext));

	/*
	 * Loop through the file, building a map of the extents.
	 * All of this is done in filesystem blocks size units.
	 *
	 * Assumptions:
	 * Throughout the file, there can be any number of blocks backed by holes
	 * or by allocated blocks.  Tail-packed files are special - if we find a file
	 * that has a size and has no allocated blocks, we could flag it as a "tail-packed"
	 * file if we cared: data is packed into the tail space of the inode block.
	 */
	int extent_num=0;
    __u64 last_block = -4;
	for (blk_idx = 0; blk_idx < num_blocks; blk_idx++) {
		unsigned int blknum = blk_idx;
		__u64 blknum64;
		/*
		 * FIBMAP takes a block index as input and on return replaces it with a
		 * block number relative to the beginning of the filesystem/partition.
		 * An output value of zero means "unallocated", or a "hole" in a sparse file.
		 * Note that this is a 32-bit value, so it will not work properly on
		 * files/filesystems with more than 4 billion blocks (~16TB),
		 */
		if (ioctl(fd, FIBMAP, &blknum) == -1) {
			int err = errno;
			perror("ioctl(FIBMAP)");
			return err;
		}
		blknum64 = blknum;	/* work in 64-bits as much as possible */
        if(blknum == 0)
        {
            printf("[%d] has unallocated block\n", fd);
            return 0;
        }

		if (blk_idx && blknum64 == (last_block + 1)) {
			/*
			 * Continuation of extent: Bump last_block and block_count.
			 */
			//ext.last_block = blknum64 ? blknum64 : hole;
			ext.block_count += FS_BLKSIZE/SECTORSIZE;
		} else {
			/*
			 * New extent: print previous extent (if any), and re-init the extent record.
			 */
			if (blk_idx){
                result.push_back(ext);
			
			}
			last_block = -4;
			ext.first_block = blknum64;
			//ext.last_block  = blknum64 ? blknum64 : hole;
			ext.block_count = FS_BLKSIZE/SECTORSIZE;
			ext.byte_offset = blk_idx * blksize;
		}
        last_block = blknum64;
	}
	result.push_back(ext);
	return (int)result.size();
}

//记录文件的name, lba
char current_path[512];
//extern "C" {
//TODO:计算文件系统开销需要使用的内存大概需要多少
int produce_fsinfo(const char *path, int depth)
{
    DIR *p;
    struct dirent *entry;
    struct stat statbuf;
    char child[512], cur[512];
    int extent_num;
    int fd;
    
    // if (fatherinode == 0 && depth == 0)
    // {
    //     file_lbas_map.clear();
    //     if ((p = opendir(path)) == NULL)
    //     {
    //         printf("open directory in failure\n");
    //         return -1;
    //     }
    //     entry = readdir(p);
    //     while ((entry = readdir(p)) != NULL){
    //         if (strcmp(entry->d_name, ".") == 0){
    //             fatherinode = entry->d_ino;
    //             break;
    //         }
    //     }
    //     closedir(p);
    //     strcpy(current_path, path);
    // }
    if ((p = opendir(path)) == NULL)
    {
        printf("open directory in failure\n");
        return -1;
    }

    while ((entry = readdir(p)) != NULL)
    {
        std::vector<file_extent> lba;
        if (entry->d_type & DT_DIR)
        { //目录文件
            //if(statbuf.st_mode & S_IFDIR) printf("%s is a link file\n",entry->d_name);
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            sprintf(child, "%s/%s", path, entry->d_name);
            fd = open(child, O_RDONLY);
            fstat(fd, &statbuf);
            //filename, inode, LBAlist, fatherinode, version=1
            produce_fsinfo(child, depth + 1);
        }
        //if(entry->d_type & ){
        else
        {
            sprintf(cur, "%s/%s", path, entry->d_name);
            fd = open(cur, O_RDONLY);
            fstat(fd, &statbuf);
            //get LBAlist
            extent_num = walk_fibmap(fd, &statbuf, FS_BLKSIZE, FS_BLKSIZE / SECTORSIZE, 0, lba); 
            if(extent_num > 0)
                file_lbas_map[cur] = std::make_pair(statbuf.st_size,lba);
            close(fd);
        }
    }
    closedir(p);
   
    return 0;
}
//};