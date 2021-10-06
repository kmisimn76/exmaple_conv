CC = g++
CXXFLAGS = -std=c++11 -O3 -Wunused-result -I/usr/local/cuda/include
LDFLAGS = -lz -L/usr/local/cuda/lib64 -lOpenCL

DEBUGFLAGS = -g #-DDEBUG

all: run.debug run.exe

run.debug: main.cpp
	$(CC) $(DEBUGFLAGS) -o run.debug main.cpp $(CXXFLAGS) $(LDFLAGS)

run.exe: main.cpp
	$(CC) -o run.exe main.cpp $(CXXFLAGS) $(LDFLAGS)

%: %.cpp
	$(CC) -o $@ $?  $(CXXFLAGS) $(LDFLAGS)

clean:
	rm run.*
