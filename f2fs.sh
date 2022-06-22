DEV="/dev/nvme0n1 /dev/nvme1n1"
DEV_MIRROR=/dev/nvme0n1 #SPDK use
MOUNT_DIR=/mnt/f2fs
file_list="./lib/nvmf/data/*.pb ./lib/nvmf/data/*.dat /home/femu/pic/*.bmp"
face_detecion_list="/home/femu/pic/test.bmp"
dds_file="/home/femu/cuda-samples/Samples/dxtc/data/lena_ref.dds"
ppm_file="/home/femu/cuda-samples/Samples/dxtc/data/lena_std.ppm"
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

    if [[ $# -eq 1 ]]; then
        if [[ $1 -eq 0 ]]; then
            for i in $(seq 1 90)  
            do
                sudo cp $face_detecion_list $dst_dir"/test_"$i".bmp"
                sudo cp $dds_file $dst_dir"/test_"$i".dds"
                sudo cp $ppm_file $dst_dir"/test_"$i".ppm"
            done
	    sudo cp -r  /home/femu/spdk/ $dst_dir
	    sudo cp -r /home/femu/spdk $dst_dir/ll
        fi
    else
        echo "invalid file system option"
    fi

}


function help(){
    echo "$0 file system type --- (0:face 1:search)"
    exit
}

if [ $# -eq 0 ]; then
    help
fi

#必须要两块盘
for dev in ${DEV[*]}
do
echo "current device: $dev"
sudo mkfs.f2fs $dev 
sudo mount $dev $MOUNT_DIR
copy_files $1
sync
sudo umount $MOUNT_DIR
done
exit

echo "@@@copy file system to another..."
sudo dd if=$DEV of=$DEV_MIRROR bs=2M  count=2048
echo "@@@file system is successfully made"

