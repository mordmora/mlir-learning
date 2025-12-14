# tools/automata-opt/CMakeLists.txt
add_llvm_executable(automata-opt
  automata-opt.cpp
)

target_link_libraries(automata-opt PRIVATE
  AutomataLib
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRMlirOptMain
)