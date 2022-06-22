APP = mmod face_project

LDFLAGS := -Wl,-rpath,/usr/local/cuda-11.5/lib64 ../dlib/build/dlib/libdlib.a \
/usr/local/cuda-11.5/lib64/libcudart_static.a -lpthread -ldl -lrt -lpthread -ldl -lrt -lX11 \
 /usr/local/cuda-11.5/lib64/libcublas.so /usr/local/cuda-11.5/lib64/libcudnn.so \
 /usr/local/cuda-11.5/lib64/libcurand.so /usr/local/cuda-11.5/lib64/libcusolver.so \
 /usr/local/cuda-11.5/lib64/libcudart.so -pthread `pkg-config opencv4 --libs` \
 -I ../dlib/ 

CFLAGS := -O3 -DNDEBUG -DDLIB_NO_GUI_SUPPORT -DDLIB_USE_CUDA `pkg-config opencv4 --cflags`


all: $(APP)
mmod: mmod.cpp
	/usr/bin/c++ $(CFLAGS)  mmod.cpp -o $@ $(LDFLAGS)

#$(CFLAGS)
face_project: face_detection.cpp
	nvcc  -DDEBUG -g -lpthread  -lX11 face_detection.cpp -o $@ `pkg-config opencv4 --cflags --libs`  -ldlib
	
#-I ../dlib ../dlib/dlib/all/source.cpp
