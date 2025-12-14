// lib/Automata/AutomataDialect.cpp
#include "automata/AutomataDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::automata;

#include "automata/AutomataDialect.cpp.inc"

#define GET_OP_CLASSES
#include "automata/AutomataOpsImpl.inc"

void AutomataDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "automata/AutomataOpsImpl.inc"
    >();
}
