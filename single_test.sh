sudo nvme list | grep SPDK
#echo $?
#exit
if [[ $? -eq  1 ]]; then
	./nvmf_create_bdev.sh 
	./local_connect.sh
fi
./send_command.sh ./decompress.json

