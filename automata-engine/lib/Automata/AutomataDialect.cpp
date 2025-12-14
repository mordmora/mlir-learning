// lib/Automata/AutomataDialect.cpp
#include "automata/AutomataDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::automata;

// ==========================================================
// SECCIÓN 1: Implementación del DIALECTO
// ==========================================================
#include "automata/AutomataDialect.cpp.inc"

// ==========================================================
// SECCIÓN 2: Implementación de las OPERACIONES
// ==========================================================
// ¡OJO! Este define DEBE ir justo antes del include de Ops.cpp.inc
#define GET_OP_CLASSES
#include "automata/AutomataOpsImpl.inc"

// ==========================================================
// SECCIÓN 3: Inicialización
// ==========================================================
void AutomataDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "automata/AutomataOpsImpl.inc"
    >();
}