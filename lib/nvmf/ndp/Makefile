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

SPDK_ROOT_DIR := $(abspath $(CURDIR)/../../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk

SO_VER := 10
SO_MINOR := 0

#CXX_SRCS = file_info.cpp

LIBNAME = ndp_env
ifeq ($(CONFIG_NDP), y)
NVCCFLAGS += -Indp  `pkg-config opencv4 --cflags` -MMD -MF $*.d.tmp
#单独编译图像处理文件
LDFLAGS += -L/usr/local/cuda-11.5/lib64 -lcudart -Idlib dlib/dlib/all/source.cpp -lpthread -lX11 `pkg-config opencv4  --libs` 
endif

#echo "config ndp $(CUDA_FLAGS)"

#CFLAGS += -lcudart -L/usr/local/cuda-11.5/lib64 #TODO ADD ndp CHOISE
#LDFLAGS += -L/usr/local/cuda-11.5/lib64 -lcudart `pkg-config opencv4 --libs`#_static

SPDK_MAP_FILE = $(abspath $(CURDIR)/spdk_nvmf.map)

include $(SPDK_ROOT_DIR)/mk/spdk.lib.mk
