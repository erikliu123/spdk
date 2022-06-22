#
#  BSD LICENSE
#
#  Copyright (c) Intel Corporation.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution.
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk

SO_VER := 10
SO_MINOR := 0

C_SRCS = ctrlr.c ctrlr_discovery.c ctrlr_bdev.c \
	 subsystem.c nvmf.c nvmf_rpc.c transport.c tcp.c

#CUDA_SRCS-$(CONFIG_NDP) = file_info.cu #不这么使用, 但是为什么还是加了两次呢
C_SRCS-$(CONFIG_RDMA) += rdma.c
LIBNAME = nvmf
LOCAL_SYS_LIBS =

CUDA_SRCS = file_info.cu ndp.cu ndp_cv.cu
#DIRS-$(CONFIG_NDP) += ndp #当前有问题，子目录肯定是更好的方式

CFLAGS += $(shell pkg-config --cflags json-c)
LDFLAGS += $(shell pkg-config --libs json-c)


ifeq ($(CONFIG_RDMA),y)
LOCAL_SYS_LIBS += -libverbs -lrdmacm
#Attach only if FreeBSD and RDMA is specified with configure
ifeq ($(OS),FreeBSD)
# Mellanox - MLX4 HBA Userspace Library
ifneq ("$(wildcard /usr/lib/libmlx4.*)","")
LOCAL_SYS_LIBS += -lmlx4
endif
# Mellanox - MLX5 HBA Userspace Library
ifneq ("$(wildcard /usr/lib/libmlx5.*)","")
LOCAL_SYS_LIBS += -lmlx5
endif
# Chelsio HBA Userspace Library
ifneq ("$(wildcard /usr/lib/libcxgb4.*)","")
LOCAL_SYS_LIBS += -lcxgb4
endif
endif
endif

ifeq ($(CONFIG_VFIO_USER), y)
C_SRCS += vfio_user.c
CFLAGS += -I$(VFIO_USER_INCLUDE_DIR)
LDFLAGS += -L$(VFIO_USER_LIBRARY_DIR)
LOCAL_SYS_LIBS += -Wl,-Bstatic -lvfio-user -Wl,-Bdynamic -ljson-c
endif

ifeq ($(CONFIG_FC),y)
C_SRCS += fc.c fc_ls.c
CFLAGS += -I$(CURDIR) $(ENV_CFLAGS)
ifneq ($(strip $(CONFIG_FC_PATH)),)
CFLAGS += -I$(CONFIG_FC_PATH)
endif
endif

ifeq ($(CONFIG_NDP), y)
#LIBS += -lspdk_ndp_env

CXXFLAGS =  -g -I../../include  -Indp -I dlib `pkg-config opencv4 --cflags` #-DDLIB_USE_CUDA #dlib/dlib/all/source.cpp  #-rdynamic不影响 -DDLIB_NO_GUI_SUPPORT 
NVCCFLAGS += -g -I../../include -Indp -Idlib `pkg-config opencv4 --cflags` -MMD -MF $*.d.tmp 
NVCCFLAGS += -DNDEBUG -DDLIB_NO_GUI_SUPPORT -DDLIB_USE_CUDA 
#单独编译图像处理文件
#LDFLAGS += -L/usr/local/cuda-11.5/lib64 -lcudart -Idlib dlib/dlib/all/source.cpp -lpthread -lX11 `pkg-config opencv4  --libs` #-Wl,-rpath,/usr/local/cuda-11.5/lib64 ../dlib/build/dlib/libdlib.a /usr/local/cuda-11.5/lib64/libcudart_static.a -lpthread -ldl -lrt -lpthread -ldl -lrt -lX11 /usr/local/cuda-11.5/lib64/libcublas.so /usr/local/cuda-11.5/lib64/libcudnn.so /usr/local/cuda-11.5/lib64/libcurand.so /usr/local/cuda-11.5/lib64/libcusolver.so /usr/local/cuda-11.5/lib64/libcudart.so -pthread `pkg-config opencv4 --cflags --libs`  
endif

all: $(DIRS-y)
	$(Q)echo "sasdsddsasaddasscan0..." ;

clean: $(DIRS-y)

include $(SPDK_ROOT_DIR)/mk/spdk.subdirs.mk

#echo "config ndp $(CUDA_FLAGS)"

#CFLAGS += -lcudart -L/usr/local/cuda-11.5/lib64 #TODO ADD ndp CHOISE
#LDFLAGS += -L/usr/local/cuda-11.5/lib64 -lcudart `pkg-config opencv4 --libs`#_static

SPDK_MAP_FILE = $(abspath $(CURDIR)/spdk_nvmf.map)

include $(SPDK_ROOT_DIR)/mk/spdk.lib.mk
