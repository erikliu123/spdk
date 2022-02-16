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
include $(SPDK_ROOT_DIR)/mk/spdk.modules.mk

APP = nvmf_tgt

C_SRCS := nvmf_main.c

SPDK_LIB_LIST = $(ALL_MODULES_LIST) event event_nvmf

ifeq ($(SPDK_ROOT_DIR)/lib/env_dpdk,$(CONFIG_ENV))
SPDK_LIB_LIST += env_dpdk_rpc
endif

ifeq ($(OS),Linux)
SPDK_LIB_LIST += event_nbd
endif

ifeq ($(CONFIG_NDP), y)
#EXTRA_FLAGS = -L/usr/local/cuda-11.5/lib64 -lcudart -lpthread -lX11 -I /home/femu/spdk/lib/nvmf/dlib \
	/home/femu/spdk/lib/nvmf/dlib/dlib/all/source.cpp `pkg-config opencv4 --libs`  #-Wl,-rpath,/usr/local/cuda-11.5/lib64 \
	 -ldl -lrt -lpthread -ldl -lrt  -lX11 \
	/usr/local/cuda-11.5/lib64/libcublas.so /usr/local/cuda-11.5/lib64/libcudnn.so \
	/usr/local/cuda-11.5/lib64/libcurand.so /usr/local/cuda-11.5/lib64/libcusolver.so \
	/usr/local/cuda-11.5/lib64/libcudart.so /usr/local/cuda-11.5/lib64/libcudart.so -pthread \
	

#/home/femu/spdk/lib/nvmf/dlib/build/dlib/libdlib.a
#-rdc=true

#LDFLAGS += -Wl,-rpath,/usr/local/cuda-11.5/lib64 /home/femu/spdk/lib/nvmf/dlib/build/dlib/libdlib.a  -lpthread -ldl -lrt  -lX11 /usr/local/cuda-11.5/lib64/libcublas.so /usr/local/cuda-11.5/lib64/libcudnn.so /usr/local/cuda-11.5/lib64/libcurand.so /usr/local/cuda-11.5/lib64/libcusolver.so /usr/local/cuda-11.5/lib64/libcudart.so -pthread `pkg-config opencv4 --cflags --libs`  -I ../dlib/  -L/usr/local/cuda-11.5/lib64 -lcudart -I /home/femu/spdk/lib/nvmf/dlib /home/femu/spdk/lib/nvmf/dlib/dlib/all/source.cpp 
endif

include $(SPDK_ROOT_DIR)/mk/spdk.app.mk
