# tests/test-progs/matching_simulation_ivf/Makefile

# --- Host paths (resolved on host) ---
GEM5_ROOT := $(abspath ../../..)

# --- Container paths ---
WORKDIR_CONT := /work/tests/test-progs/ivf_matching
M5_INC_CONT  := /work/include
M5_LIB_CONT  := /work/util/m5/build/arm64/out/libm5.a

# --- Docker image ---
DOCKCROSS_IMAGE ?= dockcross/linux-arm64:latest

# --- Outputs ---
OUT_CIM  := bin/ivf_sim
OUT_TEST := ivf_bank_check

# --- Sources ---
CIM_API_SRC := src/cim_api.cpp

APP_SRCS_CIM := \
  src/ivf_main.cpp \
  src/file_utils.cpp \
  src/bin_reader.cpp \
  src/msim_json.cpp \
  src/map_loader.cpp \
  src/query_loader.cpp \
  src/layout.cpp \
  src/msim_bins.cpp \
  src/ivf_layout.cpp \
  src/ivf_map_writer.cpp \
  src/ivf_matcher.cpp

# --- Host test compiler/flags (x86 host) ---
HOST_CXX ?= g++-12
CXXFLAGS_TEST := -O2 -DNDEBUG -Iinclude -std=gnu++17

APP_SRCS_TEST := \
  src/ivf_bank_check.cpp \
  src/file_utils.cpp \
  src/bin_reader.cpp \
  src/msim_json.cpp \
  src/msim_bins.cpp \
  src/map_loader.cpp

# --- Flags ---
CXXFLAGS_COMMON := -O3 -DNDEBUG -Iinclude -I$(M5_INC_CONT) -std=gnu++17 -DSTD_PRINT_OUTPUT
CXXFLAGS_CIM    := $(CXXFLAGS_COMMON) -DinCIM=1
CXXFLAGS_TEST   := -O2 -DNDEBUG -Iinclude -std=gnu++17

LDFLAGS := -static
LDLIBS  := $(M5_LIB_CONT)

.PHONY: all m5lib cim test clean

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
	    echo "[build] ivf_sim -> $(OUT_CIM)"; \
	    $$CXX $(CXXFLAGS_CIM) $(CIM_API_SRC) $(APP_SRCS_CIM) -o $(OUT_CIM) $(LDFLAGS) $(LDLIBS)'

# host-side checker (no m5, no cim_api, no static)
test:
	@mkdir -p bin
	@echo "[build-host] ivf_bank_check -> $(OUT_TEST)"
	@$(HOST_CXX) $(CXXFLAGS_TEST) $(APP_SRCS_TEST) -o $(OUT_TEST)

clean:
	rm -rf bin