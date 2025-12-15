#include "automata/AutomataDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::automata;

struct CellUpdateLowering : public OpConversionPattern<CellUpdateOp> {
    using OpConversionPattern<CellUpdateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CellUpdateOp op, CellUpdateOpAdaptor adapt, ConversionPatternRewriter &rew) const override {
        Location loc = op.getLoc();

        Type i32 = rew.getI32Type();

        Value up = op.getUp();
        Value right = op.getRight();
        Value down = op.getDown();
        Value left = op.getLeft();

        //ssa

        Value s1 = rew.create<arith::AddIOp>(loc, right, left);
        Value s2 = rew.create<arith::AddIOp>(loc, up, down);
        Value s3 = rew.create<arith::AddIOp>(loc, s1, s2);

        Value const2 = rew.create<arith::ConstantIntOp>(loc, i32, 2);

        Value mod2reminder = rew.create<arith::RemSIOp>(loc, s3, const2);

        rew.replaceOp(op, mod2reminder);
        
        return success();
    
    }

};


struct Automata2ArithPass : public PassWrapper<Automata2ArithPass, OperationPass<ModuleOp>>{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Automata2ArithPass);

    llvm::StringRef getArgument() const override {
        return "convert-automata-to-arith";
    }

    llvm::StringRef getDescription() const override {
        return "convierte cell_update a operaciones aritmeticas";
    }

    void getDependentDialects(DialectRegistry &reg) const override {
        reg.insert<arith::ArithDialect>();
    }

    void runOnOperation() override {
        ConversionTarget target(getContext());

        target.addLegalDialect<arith::ArithDialect>();
        target.addIllegalOp<CellUpdateOp>();

        RewritePatternSet patterns(&getContext());
        patterns.add<CellUpdateLowering>(&getContext());

        if(failed(applyPartialConversion(getOperation(), target, std::move(patterns)))){
            signalPassFailure();
        }
        
    }
};


#include "automata/AutomataPass.h"

namespace mlir {
    namespace automata{

        std::unique_ptr<mlir::Pass> createAutomata2ArithPass(){
            return std::make_unique<Automata2ArithPass>();
        }

        void registerAutomataPasses(){
            PassRegistration<Automata2ArithPass>();
        }

    }
}