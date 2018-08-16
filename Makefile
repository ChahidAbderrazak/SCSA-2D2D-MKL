include make.inc

# include and lib paths
INCLUDES=-I${_MKL_ROOT_}/include #-I/home/chahida/Work_space/Sparse_Solver/FILTLAN/INCLUDE/
LIB_PATH= -L${_MKL_ROOT_}/lib/em64t #-L/home/chahida/Work_space/Sparse_Solver/FILTLAN/LIB/

# libraries to link against
#LIBS= -lnvblas -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lcudart -lcublas -lcholinv -lstdc++ -liomp5 -lpthread -lm
#LIBS= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lcudart -lcublas -lcholinv -lstdc++ -liomp5 -lpthread -lm
LIBS= -mkl=parallel
#LIBS= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core #-ldfiltlan
LIBS+= -lstdc++ -liomp5 -lpthread -lm

CPP_SRC=SCSA_2D2D_MKL.cpp

ALL_OBJ=$(CPP_SRC:.cpp=.o)

EXE=$(CPP_SRC:.cpp=)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

SCSA_%: SCSA_%.o
	$(CC) $< -o $@ $(LIB_PATH) $(LIBS)

all: $(EXE)

$(EXE): $(ALL_OBJ)
        
clean:
	rm -f *.o $(EXE)
