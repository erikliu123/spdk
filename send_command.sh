DEV=/dev/nvme0n1
INPUT=./face.json


if [ $# -eq 1 ]; then
    INPUT=$1
fi
echo -e "input file: $INPUT\n\n"
#exit
ID=`cat $INPUT | grep ID | awk '{print $2}'`
ID=${ID%?} #remove last comma--','
echo "TASK ID: $ID"

function offload_task()
{
    sudo nvme io-passthru $DEV --opcode=0x81 --namespace-id=0x1 --data-len=8192 --cdw12=15 --cdw10=16 --cdw14=2   -i $INPUT --write 
    if [ $? -eq 0 ]; then
        sudo nvme io-passthru $DEV --opcode=0x82 --namespace-id=0x1  --cdw12=0 --cdw10=16 --cdw13=$ID
    else
        echo "error command"
    fi
}

offload_task

