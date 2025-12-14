# lib/Automata/CMakeLists.txt
add_mlir_library(AutomataLib
  AutomataDialect.cpp

  DEPENDS
  MLIRAutomataIncGen


  LINK_LIBS
  MLIRIR 
  MLIRSideEffectInterfaces
)