#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "automata/AutomataDialect.h.inc"

#define GET_OP_CLASSES
#include "automata/AutomataOps.h.inc"