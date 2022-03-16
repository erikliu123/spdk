bdev=Malloc0
subsytem=nqn.2016-06.io.spdk:cnode2
IP=127.0.0.1
PORT=4120

sudo  cat <<EOF
bdev_nvme_attach_controller -b $bdev -t PCIe -a 0000:00:05.0
EOF

sudo scripts/rpc.py <<EOF
bdev_nvme_attach_controller -b $bdev -t PCIe -a 0000:00:05.0
nvmf_create_subsystem $subsytem -a -s SPDK00000000000002 -m 10 
nvmf_create_transport -t tcp
nvmf_subsystem_add_listener $subsytem -t tcp -a $IP -s $PORT
nvmf_subsystem_add_ns $subsytem $bdev"n1"
EOF

sudo modprobe nvme_tcp
HOSTNQN=`sudo nvme gen-hostnqn`
sudo nvme connect -t tcp -a $IP -s $PORT -n  "$subsytem" --hostnqn=$HOSTNQN

echo "list nvme devices..."
sudo nvme list
