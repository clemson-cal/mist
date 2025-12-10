.PHONY: all examples tests clean

all: examples tests

examples:
	$(MAKE) -C examples/advection-1d
	$(MAKE) -C examples/advection-1d-decomp
	$(MAKE) -C examples/advection-1d-parallel
	$(MAKE) -C examples/config-reader

tests:
	$(MAKE) -C tests

clean:
	$(MAKE) -C examples/advection-1d clean
	$(MAKE) -C examples/advection-1d-decomp clean
	$(MAKE) -C examples/advection-1d-parallel clean
	$(MAKE) -C examples/config-reader clean
	$(MAKE) -C tests clean
