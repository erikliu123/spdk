if [ $# -eq 0 ]; then	
	./mkfs.sh 0
fi
sudo ./scripts/setup.sh
sudo ./build/bin/nvmf_tgt

