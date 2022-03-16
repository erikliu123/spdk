DEV=/dev/nvme0
sudo nvme io-passthru $DEV --opcode=0x81 --namespace-id=0x1 --data-len=8192 --cdw12=15 --cdw10=16 --cdw14=2   -i /home/femu/spdk/face.json --write 
sudo nvme io-passthru $DEV --opcode=0x82 --namespace-id=0x1  --cdw12=0 --cdw10=16 --cdw13=1  
