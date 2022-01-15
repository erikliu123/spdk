#ifndef __FILE_INFO__
#define __FILE_INFO__
#define FS_BLKSIZE (4096)
#define SECTORSIZE (512)
#include <iostream>
#include <vector>
struct file_extent {
	__u64 byte_offset;
	__u64 first_block;
	__u64 block_count;
};


// extern "C"
// {
// int produce_fsinfo(const char *path, int fatherinode, int depth);
// };
int produce_fsinfo(const char *path, int depth);

#endif