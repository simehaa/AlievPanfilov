POPC = popc
CXX = g++-11
CPPFLAGS = -std=c++20
CXXFLAGS = -O3
POPXXFLAGS = -O3 -target=ipu2
LDLIBS = -lpoplar -lboost_program_options
TARGETS = main codelets.gp
SOURCES = main.cpp
SRCDIR = src
INCDIR = -I./include
SRC := $(wildcard $(SRCDIR)/*.cpp)
OBJ	:= $(SRC:$(SRCDIR)/%.cpp=%.o)

all: $(TARGETS)

# Primary binary: compilation
main: $(OBJ)
	$(CXX) $+ $(LDLIBS) -o $@

# Object file and dependencies
.INTERMEDIATE: $(OBJ)
%.o: $(SRCDIR)/%.cpp
	$(CXX) $(INCDIR) -c $+ $(LDLIBS) -o $@

# Codelet compilation with popc
%.gp: %.cpp
	$(POPC) $(POPXXFLAGS) $+ -o $@

.PHONY: clean
clean:
	$(RM) $(TARGETS)
