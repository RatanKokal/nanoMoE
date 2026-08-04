# bench/cpp/Makefile
# Builds the CPU-side BlockAllocator micro-benchmark (no CUDA dependency).

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -pthread -march=native -Wall
SRCS     := bench_allocator.cpp ../../src/mem_manager.cpp
TARGET   := bench_alloc

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
