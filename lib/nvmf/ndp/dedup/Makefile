APP=dedup

dedup: md5.cpp  md5kernel.cu deduplication.cpp
	nvcc $^ -o  $(APP)

clean:
	rm $(APP)
