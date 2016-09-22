SHELL := /bin/bash

ifndef CXX
CXX := g++
endif

ifndef PY
PY := python3
endif

CPP_INCLUDE := -I./src/
CPP_LINK    := -ldl -pthread

all: src/meta pymodules

src/meta: src/meta.cpp
	$(CXX) -std=c++1z $(CPP_LINK) $(CPP_INCLUDE) $^ -o $@

pymodules:
	$(PY) setup.py develop --user

clean:
	rm -rf $(CPP_OBJECTS)
	rm -rf `find . -name "*.o"`
	rm -rf `find . -name "*.so"`

.PHONY: all pymodules clean

# .NOTPARALLEL
