#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "automata/AutomataDialect.h"

int main(int argc, char** argv){

    mlir::DialectRegistry reg;
    mlir::registerAllDialects(reg);

    reg.insert<mlir::automata::AutomataDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "automata engine opt", reg)
    );
}