

#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information
LOG_LEVEL=5

NVCC=nvcc
CXX=g++

CUDA_ROOT=$(shell dirname `which nvcc`)/..
UAMMD_ROOT=../../../GrainSimTorquesv2
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
INCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
# INCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src/third_party
NVCCFLAGS=-g -ccbin=$(CXX) -std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) --extended-lambda --expt-relaxed-constexpr

# New: allow selecting files/target from command line
# Provide FILES="a.cu b.cu" or TARGET=name
# FILES ?= ShearGranularMedia.cu
# TARGET ?= app

# ifeq ($(TARGET),)
#   SRCS := $(if $(FILES),$(wildcard $(FILES)),$(wildcard *.cu))
#   TARGETS := $(patsubst %.cu, %, $(SRCS))
# else
#   SRCS := $(TARGET).cu
#   TARGETS := $(TARGET)
# endif

all: $(patsubst %.cu, %, $(wildcard *.cu))
# all: $(TARGETS)

%: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $(@:.out=)


12-%: 12-%.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $(@:.out=) -lcufft

13-%: 13-%.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $(@:.out=) -lcufft

clean: $(patsubst %.cu, %.clean, $(wildcard *.cu))

%.clean:
	rm -f $(@:.clean=)
