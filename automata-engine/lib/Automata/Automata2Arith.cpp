#include "automata/AutomataDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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

struct EvolveLowering : public OpConversionPattern<EvolveOp> {

    using OpConversionPattern<EvolveOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(EvolveOp op, EvolveOpAdaptor adapt, ConversionPatternRewriter& rew) const override {

        Location loc = op.getLoc();

        auto array_in = op.getInArray();
        auto array_out = op.getOutArray();

        Value constant_dim1_idx = rew.create<arith::ConstantIndexOp>(loc, 0);
        memref::DimOp dOp_dim1 = rew.create<memref::DimOp>(loc, op.getInArray(), constant_dim1_idx);
        Value size_d1 = dOp_dim1.getResult();

        Value constant_dim2_idx = rew.create<arith::ConstantIndexOp>(loc, 1);
        memref::DimOp dOp_dim2 = rew.create<memref::DimOp>(loc, op.getInArray(), constant_dim2_idx);
        Value size_d2 = dOp_dim2.getResult();

        //Value tot_siz = rew.create<arith::AddIOp>(loc, size_d1, size_d2);

        using ConvPattRew = ConversionPatternRewriter;

        Value zero = rew.create<arith::ConstantIndexOp>(loc, 0);        
        Value one = rew.create<arith::ConstantIndexOp>(loc, 1);

        ValueRange lbs = {zero, zero};
        ValueRange ubs = {size_d1, size_d2};
        ValueRange step = {one, one};


        scf::buildLoopNest(
            rew, loc, lbs, ubs, step, [&](OpBuilder& b, Location loc, ValueRange ivs){
                Value i = ivs[0];
                Value j = ivs[1];

                //condicion (i >= 0 && i < size_d1)
                ValueRange o1 = {i, zero};
                Value i_goet = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, o1); 

                ValueRange o2 = {i, size_d1};
                Value i_lth = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, o2);

                ValueRange o3 = {i_goet, i_lth};
                Value and_cmpi = b.create<arith::AndIOp>(loc, o3);

                //rew.create<scf::IfOp>(loc, )
                //condicion if (j >= 0 && i < size_d2)

                ValueRange o4 = {j, zero};
                Value j_goet = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, o4); 

                ValueRange o5 = {j, size_d2};
                Value j_lth = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, o5);

                ValueRange o6 = {j_goet, j_lth};
                Value and_cmpj = b.create<arith::AndIOp>(loc, o6);

                //Condicion (j >= 0 && i < size_d2) && (i >= 0 && i < size_d1)

                ValueRange o7 = {and_cmpi, and_cmpj};
                Value final_cmp = b.create<arith::AndIOp>(loc, o7);

                b.create<scf::IfOp>(loc, final_cmp, [&](OpBuilder& bldr, Location loc){

                    //quedé por acá :p

                });


            }
        );


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