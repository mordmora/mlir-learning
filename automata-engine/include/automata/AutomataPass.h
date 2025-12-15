#ifndef AUTOMATA_PASS
#define AUTOMATA_PASS

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir{
    namespace automata{
        std::unique_ptr<mlir::Pass> createAutomata2ArithPass();

        void registerAutomataPasses();
    } //namespace automata
}//namespace mlir


#endif