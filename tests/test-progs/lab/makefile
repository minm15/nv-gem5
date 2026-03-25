# tests/test-progs/lab/Makefile

# --- Host paths (resolved on host) ---
GEM5_ROOT := $(abspath ../../..)

# --- Container paths ---
WORKDIR_CONT := /work/tests/test-progs/lab
M5_INC_CONT  := /work/include
M5_LIB_CONT  := /work/util/m5/build/arm64/out/libm5.a

# --- Docker image ---
DOCKCROSS_IMAGE ?= dockcross/linux-arm64:latest

# --- Output ---
OUT := bin/hello64-static

# --- Sources ---
CIM_API_SRC := src/cim_api.cpp
CIM_SRC     := src/desc_test.cpp
CPU_SRC     := src/desc_test.cpp

# --- Flags (fixed O3) ---
CXXFLAGS_COMMON := -O3 -DNDEBUG -Iinclude -I$(M5_INC_CONT) -std=gnu++17 -DSTD_PRINT_OUTPUT
CXXFLAGS_CIM    := $(CXXFLAGS_COMMON) -DinCIM=1
CXXFLAGS_CPU    := $(CXXFLAGS_COMMON)

LDFLAGS := -static
LDLIBS  := $(M5_LIB_CONT)

.PHONY: all m5lib cim cpu clean

all: cim

# Build libm5.a inside container if missing
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

cim:
	docker run --rm \
	  -v "$(GEM5_ROOT)":/work \
	  -w $(WORKDIR_CONT) \
	  $(DOCKCROSS_IMAGE) \
	  bash -lc 'set -e; mkdir -p bin; $(ENSURE_M5LIB); \
	    echo "[build] CIM -> $(OUT)"; \
	    $$CXX $(CXXFLAGS_CIM) $(CIM_API_SRC) $(CIM_SRC) -o $(OUT) $(LDFLAGS) $(LDLIBS)'

cpu:
	docker run --rm \
	  -v "$(GEM5_ROOT)":/work \
	  -w $(WORKDIR_CONT) \
	  $(DOCKCROSS_IMAGE) \
	  bash -lc 'set -e; mkdir -p bin; $(ENSURE_M5LIB); \
	    echo "[build] CPU -> $(OUT)"; \
	    $$CXX $(CXXFLAGS_CPU) $(CPU_SRC) -o $(OUT) $(LDFLAGS) $(LDLIBS)'

clean:
	rm -rf bin