# nanoMoE  -  Top-level Makefile
#
# Delegates to the sub-Makefiles for C++ targets.
# Python tests and benchmarks require the CUDA extension to be built first
# via `pip install -e .` (needs a GPU host).
#
# Usage:
#   make test           -  build + run C++ test suite
#   make tsan           -  build + run tests under ThreadSanitizer
#   make bench          -  build + run CPU allocator benchmark
#   make build-ext      -  build the CUDA PyTorch extension (GPU required)
#   make clean          -  remove all build artefacts

.PHONY: test tsan bench build-ext clean

test:
	$(MAKE) -C tests/cpp run

tsan:
	$(MAKE) -C tests/cpp tsan
	cd tests/cpp && ./test_alloc_tsan

bench:
	$(MAKE) -C bench/cpp run

build-ext:
	pip install -e . --no-build-isolation

clean:
	$(MAKE) -C tests/cpp clean
	$(MAKE) -C bench/cpp clean
	rm -rf build dist *.egg-info
