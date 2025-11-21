CXX = c++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -I../include
TARGET = test_serialize

all: $(TARGET)
	./$(TARGET)

$(TARGET): test_serialize.cpp \
           ../include/mist/core.hpp \
           ../include/mist/serialize.hpp \
           ../include/mist/ascii_writer.hpp \
           ../include/mist/ascii_reader.hpp \
           ../include/mist/binary_writer.hpp \
           ../include/mist/binary_reader.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) test_serialize.cpp

clean:
	rm -f $(TARGET)

.PHONY: all clean
