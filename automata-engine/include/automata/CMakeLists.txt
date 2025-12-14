# include/automata/CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS AutomataDialect.td)

mlir_tablegen(AutomataDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AutomataDialect.cpp.inc -gen-dialect-defs)

mlir_tablegen(AutomataOps.h.inc -gen-op-decls)
mlir_tablegen(AutomataOpsImpl.inc -gen-op-defs)

add_public_tablegen_target(MLIRAutomataIncGen)  