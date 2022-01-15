DEV=/dev/nvme1n1
DEV_MIRROR=/dev/nvme0n1 #SPDK use
MOUNT_DIR=/mnt/ndp
file_list="./lib/nvmf/data/*.pb ./lib/nvmf/data/*.dat /home/femu/pic/*.bmp"
dst_dir=$MOUNT_DIR

function copy_files()
{
    str=${file_list//|/ } #将读入的line以空格拆分，保存到数组
	#echo str 长度: ${#str[@]}
    arr=($str)

    for each in ${arr[*]}
    do
       sudo cp $each $dst_dir
    done
}
#必须要两块盘
sudo mkfs.ext4 $DEV 
sudo mount $DEV $MOUNT_DIR
copy_files

echo "@@@copy file system to another..."
sudo dd if=$DEV of=$DEV_MIRROR bs=2M  count=2048
echo "@@@file system is successfully made"

