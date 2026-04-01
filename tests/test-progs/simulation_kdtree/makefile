# tests/test-progs/pf_kernel_roi/Makefile

GEM5_ROOT := $(abspath ../../..)

WORKDIR_CONT := /work/tests/test-progs/simulation_kdtree
M5_INC_CONT  := /work/include
M5_LIB_CONT  := /work/util/m5/build/arm64/out/libm5.a

DOCKCROSS_IMAGE ?= dockcross/linux-arm64:latest

OUT := bin/pf_kernel_roi
SRCS := main.cpp
# SRCS := data_wo_cim.cpp

CXXFLAGS := -O1 -DNDEBUG -std=gnu++17 -DUSE_M5 \
            -I. -I$(M5_INC_CONT)
LDFLAGS  := -static
LDLIBS   := $(M5_LIB_CONT)

.PHONY: all m5lib build clean

all: build

m5lib:
	docker run --rm \
	  -v "$(GEM5_ROOT)":/work \
	  -w /work/util/m5 \
	  $(DOCKCROSS_IMAGE) \
	  bash -lc 'set -e; make -j$$(nproc) build/arm64/out/libm5.a'

define ENSURE_M5LIB
if [ ! -f "$(M5_LIB_CONT)" ]; then \
  echo "[m5] libm5.a not found -> building..."; \
  cd /work/util/m5; \
  make -j$$(nproc) build/arm64/out/libm5.a; \
  cd $(WORKDIR_CONT); \
fi
endef

build:
	docker run --rm \
	  -v "$(GEM5_ROOT)":/work \
	  -w $(WORKDIR_CONT) \
	  $(DOCKCROSS_IMAGE) \
	  bash -lc 'set -e; mkdir -p bin; $(ENSURE_M5LIB); \
	    echo "[build] pf_kernel_roi -> $(OUT)"; \
	    $$CXX $(CXXFLAGS) $(SRCS) -o $(OUT) $(LDFLAGS) $(LDLIBS)'

clean:
	rm -rf bin
