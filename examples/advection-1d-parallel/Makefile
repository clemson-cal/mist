CXX = g++
CXXFLAGS = -std=c++20 -Wall -I../../include -I/opt/homebrew/opt/readline/include -O3
LDFLAGS = -L/opt/homebrew/opt/readline/lib -lreadline
TARGET = advection-1d-parallel

all: $(TARGET)

$(TARGET): advection-1d-parallel.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) advection-1d-parallel.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
