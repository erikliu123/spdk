bdev=Malloc0
subsytem=nqn.2016-06.io.spdk:cnode2
IP=`ifconfig ens3 | grep inet\ | awk '{ print $2 }'` 
PORT=4120

echo "IP: $IP"

sudo modprobe nvme_tcp
HOSTNQN=`sudo nvme gen-hostnqn`
sudo nvme connect -t tcp -a $IP -s $PORT -n  "$subsytem" --hostnqn=$HOSTNQN

echo "list nvme devices..."
sudo nvme list

sudo mount /dev/nvme0n1 /mnt/nfs
#mount -t nfs $IP:/mnt/nfs /mnt/share


#sudo  nvme io-passthru /dev/nvme0 --opcode=0x82 --namespace-id=0x1 --data-len=8192 --cdw12=15 --cdw10=16  --read
#sudo  nvme io-passthru /dev/nvme0 --opcode=0x82 --namespace-id=0x1 --data-len=8192 --cdw12=15 --cdw10=16 --cdw14=3  --read | less
