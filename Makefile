POPC = popc
CXX = g++-11
CPPFLAGS = -std=c++20
CXXFLAGS = -O3
POPXXFLAGS = -O3 -target=ipu2
LDLIBS = -lpoplar -lboost_program_options
SRCDIR = src
OBJDIR = obj
BINDIR = bin
INCDIR = -I./include
SRC := $(wildcard $(SRCDIR)/*.cpp)
SRC := $(filter-out src/codelets.cpp, $(SRC))
OBJ	:= $(SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
TARGETS = $(BINDIR)/main $(OBJDIR)/codelets.gp

all: $(TARGETS)

# compiling main binary 
$(BINDIR)/main: $(OBJ) 
	@mkdir -p $(BINDIR)
	$(CXX) $+ $(LDLIBS) -o $@

# linking object files
.INTERMEDIATE: $(OBJ)
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(INCDIR) -c $+ $(LDLIBS) -o $@

# pre-compiling codelet using popc
$(OBJDIR)/codelets.gp: $(SRCDIR)/codelets.cpp
	@mkdir -p $(OBJDIR)
	$(POPC) $(POPXXFLAGS) $+ -o $@

.PHONY: clean
clean:
	$(RM) $(TARGETS) $(OBJ)
