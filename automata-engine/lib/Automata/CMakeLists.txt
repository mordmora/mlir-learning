# lib/Automata/CMakeLists.txt
add_mlir_library(AutomataLib
  AutomataDialect.cpp
  Automata2Arith.cpp

  DEPENDS
  MLIRAutomataIncGen

  LINK_LIBS PUBLIC
  MLIRIR 
  MLIRSideEffectInterfaces
  MLIRArithDialect
  MLIRPass
  MLIRTransforms
)