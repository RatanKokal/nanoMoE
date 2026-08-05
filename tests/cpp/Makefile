# tests/cpp/Makefile
# Builds the BlockAllocator test suite (no CUDA dependency).

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -pthread -Wall -Wextra
SRCS     := test_allocator.cpp ../../src/mem_manager.cpp
TARGET   := test_alloc
TSAN_TARGET := test_alloc_tsan

.PHONY: all tsan run clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# ThreadSanitizer build  -  detects data races under concurrent tests.
tsan: $(SRCS)
	$(CXX) $(CXXFLAGS) -O1 -fsanitize=thread $^ -o $(TSAN_TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TSAN_TARGET)
