# bench/cpp/Makefile
# Builds the CPU-side BlockAllocator micro-benchmarks (no CUDA dependency).

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -pthread -march=native -Wall
SRCS     := bench_allocator.cpp ../../src/mem_manager.cpp
TARGET   := bench_alloc

LEGACY_SRCS   := bench_legacy.cpp
LEGACY_TARGET := bench_alloc_legacy

.PHONY: all legacy compare run clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Self-contained legacy binary — no dependency on current mem_manager sources.
legacy: $(LEGACY_TARGET)

$(LEGACY_TARGET): $(LEGACY_SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Run current implementation.
run: $(TARGET)
	./$(TARGET)

# Run both back-to-back for easy comparison.
compare: $(TARGET) $(LEGACY_TARGET)
	./$(LEGACY_TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(LEGACY_TARGET)
