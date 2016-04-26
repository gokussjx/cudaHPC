SHELL=/bin/bash
MKDIR=mkdir -pv
RM=rm -fv
MAKE=make
export CC=gcc
export CXX=g++
export NVCC=nvcc
export BINDIR=$(abspath bin)
export SRCDIR=$(abspath src)
export LIBDIR=$(abspath lib)
export INCLUDE=$(abspath include)
BUILD_DIRS=$(BINDIR) $(LIBDIR)

CCWARNS=-Wall
CXXWARNS=-Wall -Wno-unknown-pragmas -Wno-unused-result
export CFLAGS=-std=gnu99 -O2 $(CCWARNS) -iquote $(INCLUDE)
export CCFLAGS=-std=c++11 -O2 $(CXXWARNS) -iquote $(INCLUDE)

# =================
# Register binary targets here then add rules at the bottom
# =================
TARGETS=standard diff #hello

# =================
# General rules
# =================
.PHONY: all clean run

all:
	@echo "===== Preparing directories... ====="
	$(MKDIR) $(BUILD_DIRS)
	@echo "===== Building targets... ====="
	$(MAKE) $(addprefix $(BINDIR)/, $(TARGETS))
	@echo "===== Done. ====="

clean:
	@echo "===== Cleaning... ====="
	$(RM) -r $(BUILD_DIRS)
	@echo "===== Done. ====="

run:
	@echo "Hello"

$(LIBDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

$(LIBDIR)/%.o: $(SRCDIR)/%.cc
	$(CXX) -c $(CCFLAGS) -o $@ $<

$(LIBDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) -c -o $@ $<

# =================
# Binary targets
# =================
$(BINDIR)/hello: driver.o
	$(NVCC) -o $@ $^

$(BINDIR)/standard: $(SRCDIR)/standard.c
	$(CC) $(CFLAGS) -o $@ $^

$(BINDIR)/diff: $(SRCDIR)/diff.c
	$(CC) $(CFLAGS) -o $@ $^
