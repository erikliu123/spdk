sudo modprobe nvme_tcp
PORT=4120
IP=`ifconfig ens3 | grep inet\ | awk '{ print $2 }'` #127.0.0.1
subsytem=nqn.2016-06.io.spdk:cnode2
HOSTNQN=`sudo nvme gen-hostnqn`
sudo nvme connect -t tcp -a $IP -s $PORT -n  "$subsytem" --hostnqn=$HOSTNQN

echo "list nvme devices..."
sudo nvme list

