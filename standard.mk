SHELL=/bin/bash
MKDIR=mkdir -pv
RM=rm -fv
MAKE=make -f standard.mk
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
TARGETS=standard diff p6p5 diffb #hello

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

$(BINDIR)/%: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -o $@ $^

# =================
# Binary targets
# =================
$(BINDIR)/hello: driver.o
	$(NVCC) -o $@ $^

