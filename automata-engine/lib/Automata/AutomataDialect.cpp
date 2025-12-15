// lib/Automata/AutomataDialect.cpp
#include "automata/AutomataDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::automata;


#include "automata/AutomataDialect.cpp.inc"

#define GET_OP_CLASSES
#include "automata/AutomataOpsImpl.inc"


struct SimplifyDeadCells : OpRewritePattern<CellUpdateOp> {

    using OpRewritePattern<CellUpdateOp>::OpRewritePattern;//constructor heredado

    LogicalResult matchAndRewrite(
        CellUpdateOp op,
        PatternRewriter &rew
    ) const override {

        llvm::APInt mid, left, up, right, down;

        if(!matchPattern(op.getMid(), m_ConstantInt(&mid)) || mid != 0) return failure();
        if(!matchPattern(op.getLeft(), m_ConstantInt(&left)) || left != 0) return failure();
        if(!matchPattern(op.getUp(), m_ConstantInt(&up)) || up != 0) return failure();
        if(!matchPattern(op.getRight(), m_ConstantInt(&right)) || right != 0) return failure();
        if(!matchPattern(op.getDown(), m_ConstantInt(&down)) || down != 0) return failure();

        auto z = rew.create<arith::ConstantIntOp>(op.getLoc(), op.getType(), 0);

        rew.replaceOp(op, z);
 
        return success();
    }

};


void AutomataDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "automata/AutomataOpsImpl.inc"
    >();
}

//void CellUpdateOp::getCanonicalizationPatterns()

void CellUpdateOp::getCanonicalizationPatterns(RewritePatternSet &res, MLIRContext *ctxt){
    res.add<SimplifyDeadCells>(ctxt);
}
