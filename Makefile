ISPC=ispc
ISPC_FLAGS=--pic -O2
ISPC_ARCH_FLAGS=--cpu=broadwell --target=avx2-i32x8
ISPC_DEBUG=-g
ARCH_FLAGS=-march=native -mtune=native
CXXFLAGS=-std=c++11 -O0 -g $(ARCH_FLAGS)

%.o:%.ispc
	$(ISPC) $(ISPC_FLAGS) $(ISPC_ARCH_FLAGS) $(ISPC_DEBUG) -o $@ $^

%.S:%.ispc
	$(ISPC) $(ISPC_FLAGS) $(ISPC_ARCH_FLAGS) --x86-asm-syntax=intel --emit-asm -o $@ $^

%.ll:%.ispc
	$(ISPC) $(ISPC_FLAGS) $(ISPC_ARCH_FLAGS) --x86-asm-syntax=intel --emit-llvm-text -o $@ $^

%.cpp:%.ispc
	$(ISPC) $(ISPC_FLAGS) --cpu=generic --target=generic-8 --emit-c++ -o $@ $^

%.S:%.cc
	$(CXX) $(CXXFLAGS) -S -o $@ $^

# %.S:%.o
# 	objdump --x86-asm-syntax=intel -Cd $^ > $@

all: main

main: im2col_onednn.o copy_1d_simd.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o *.S *.cpp main
