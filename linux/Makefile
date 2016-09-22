CC=g++
CFLAGS=-Wall -L/usr/local/lib/ -I/usr/local/include/ -lgsl -lgslcblas -pthread -lm -Ofast -march=native -funroll-loops -ffast-math -Wno-unused-result 
SOURCES=tte.cpp
EXECUTABLE=tte

all: $(EXECUTABLE) 

$(EXECUTABLE): $(SOURCES) 
	$(CC) $(CFLAGS) $(SOURCES) -o $(EXECUTABLE)

clean:
	rm *o $(EXECUTABLE) 
