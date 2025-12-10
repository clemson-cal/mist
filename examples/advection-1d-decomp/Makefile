CXX = c++
CXXFLAGS = -std=c++20 -O3 -I../../include -I/opt/homebrew/opt/readline/include
LDFLAGS = -L/opt/homebrew/opt/readline/lib -lreadline
TARGETS = advection-1d-decomp

all: $(TARGETS)

advection-1d-decomp: advection-1d-decomp.cpp ../../include/mist/core.hpp ../../include/mist/driver.hpp ../../include/mist/parallel/pipeline.hpp
	$(CXX) $(CXXFLAGS) -o advection-1d-decomp advection-1d-decomp.cpp $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.dat

.PHONY: all clean
