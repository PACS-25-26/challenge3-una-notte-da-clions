# Note by Andrea: this make file is a slightly edited corrected version of the Makefile form the first challenge. 
# to perform a new compilation and launch the executable use: make clean && make && ./executables/exe

# Set compiler and output name
CXX = mpic++
SRC = main.cpp
TARGET ?= exe

# Set directories for executable files, headers and sources
EXEDIR = executables
HHDIR = include
LIBDIR = lib

# Use enable ANSI escape sequences for colors!
COLOR ?= true

# Standard settings for compilation (Aggiunto il path di Eigen)
CXXFLAGS = -std=c++23 -O1 -I /usr/include/eigen3

# Select alpha policy
EXECUTION ?= PARALLEL
OPENMP = -fopenmp


# ----------------------

# Builds the sequence of files to be compiled together with main
SRC += $(wildcard $(LIBDIR)/*.cpp)

# Builds the sequence of object files used for the following stage
OBJ = $(SRC:.cpp=.o)

# Set these actions as phony
.PHONY: clean debug release all

# Linking: creates an executable file from the object files. It also creates a directory for it
$(TARGET): $(OBJ)
	mkdir -p $(EXEDIR) 
	$(CXX) $(CXXFLAGS) $(OPENMP) -o $(EXEDIR)/$@ $^ $(LDFLAGS)

# Compilation: builds .o file from the correspoinding .cpp 
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(HHDIR) -c $< -o $@ -DEXECUTION=$(EXECUTION) -DCOLOR=$(COLOR) $(OPENMP)

# Clears current .o files to allow a new compilation. Used when building a debug o release version
clean:
	find . -name '*.o' -type f -delete

# Debug mode. It shows more warnings and lowers optimization
debug: CXXFLAGS = -std=c++23 -Wall -g -O0 -I /usr/include/eigen3
debug: all

# Release mode. Adds optimization and removes debug features 
release: CXXFLAGS = -std=c++23 -O3 -DNDEBUG -I /usr/include/eigen3
release: all

# Launches the compilation process
all: $(TARGET)