CXX = c++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -I../include
TARGETS = test_serialize test_ndarray

all: $(TARGETS)
	./test_serialize
	./test_ndarray

test_serialize: test_serialize.cpp \
                ../include/mist/core.hpp \
                ../include/mist/serialize.hpp \
                ../include/mist/ascii_writer.hpp \
                ../include/mist/ascii_reader.hpp \
                ../include/mist/binary_writer.hpp \
                ../include/mist/binary_reader.hpp
	$(CXX) $(CXXFLAGS) -o test_serialize test_serialize.cpp

test_ndarray: test_ndarray.cpp \
              ../include/mist/core.hpp \
              ../include/mist/ndarray.hpp
	$(CXX) $(CXXFLAGS) -o test_ndarray test_ndarray.cpp

clean:
	rm -f $(TARGETS)

.PHONY: all clean
