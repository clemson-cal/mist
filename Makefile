.PHONY: all examples tests clean

all: examples tests

examples:
	$(MAKE) -C examples/advection-1d
	$(MAKE) -C examples/config-reader

tests:
	@echo "Building tests..."
	c++ -std=c++20 -Wall -Wextra -O2 -I include -o tests/test_serialize tests/test_serialize.cpp
	@echo "Running tests..."
	./tests/test_serialize

clean:
	$(MAKE) -C examples/advection-1d clean
	$(MAKE) -C examples/config-reader clean
	rm -f tests/test_serialize
