# Usage:
#   make SRC=my_kernel.cpp TARGET=my_kernel
#
# Example:
#   make SRC=GQA_bkwd_asm_interleaved.cpp TARGET=tk_kernel_bkwd

# Compiler
GPU_TARGET ?= CDNA4

# parameters
ATTN_B ?= 16
ATTN_H ?= 64
ATTN_H_KV ?= 8
ATTN_N ?= 8192

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
ROCM_INCLUDE_DIR := $(ROCM_INSTALL_DIR)/include
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Check that SRC and TARGET were provided
ifeq ($(strip $(SRC)),)
  $(error You must provide SRC=<source_file.cpp>)
endif
ifeq ($(strip $(TARGET)),)
  $(error You must provide TARGET=<binary_name>)
endif

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
  HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
  HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
  HIPFLAGS+= -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra
CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

ICXXFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
             $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -Rpass-analysis=kernel-resource-usage

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)*

.PHONY: all clean
