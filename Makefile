default:
	g++  -O9 -std=c++14 -O3 -o main.o main.cpp Sum.cpp NueralNetFloat.cpp NueralNetController.cpp
clean:
	rm -rfv *.o
