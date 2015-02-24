# -*- Makefile -*-

# by default, "arch" is unknown, should be specified in the command line
arch = UNKNOWN
KOKKOS_PATH = ../../../trilinos/packages/kokkos

setup_file = ../setup/Make.$(arch)
include $(setup_file)

include $(KOKKOS_PATH)/Makefile.kokkos
EXTRA_INC = $(KOKKOS_INC)
EXTRA_LIB = $(KOKKOS_LINK)

HPCG_DEPS = src/CG.o src/CG_ref.o src/TestCG.o src/ComputeResidual.o \
         src/ExchangeHalo.o src/GenerateGeometry.o src/GenerateProblem.o \
	 src/OptimizeProblem.o src/ReadHpcgDat.o src/ReportResults.o \
	 src/SetupHalo.o src/TestSymmetry.o src/TestNorms.o src/WriteProblem.o \
         src/YAML_Doc.o src/YAML_Element.o src/ComputeDotProduct.o \
         src/ComputeDotProduct_ref.o src/finalize.o src/init.o src/mytimer.o src/ComputeSPMV.o \
         src/ComputeSPMV_ref.o src/ComputeSYMGS.o src/ComputeSYMGS_ref.o src/ComputeWAXPBY.o src/ComputeWAXPBY_ref.o \
         src/ComputeMG_ref.o src/ComputeMG.o src/ComputeProlongation_ref.o src/ComputeRestriction_ref.o src/GenerateCoarseProblem.o

bin/xhpcg: kokkos_depend.o src/main.o $(HPCG_DEPS)
	#ar cr libkokkoscore.a $(OBJ_KOKKOS_LINK)
	#touch kokkos_depend.cpp
	#$(CXX) $(CXXFLAGS) $(SHFLAGS) $(EXTRA_INC) -c kokkos_depend.cpp
	$(LINKER) $(LINKFLAGS) main.o $(HPCG_DEPS) -o bin/xhpcg $(HPCG_LIBS)

%.o:%.cpp
	$(CXX) $(CXXFLAGS) $(EXTRA_INC) -c $<

clean:
	rm -f $(HPCG_DEPS) bin/xhpcg src/main.o
	rm *.o *.d libkokkoscore.a kokkos_depend.cpp *.cuda *.host

.PHONY: clean

