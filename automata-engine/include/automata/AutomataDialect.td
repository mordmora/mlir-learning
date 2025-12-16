//include/automata/AutomataDialect.td
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Automata_Dialect : Dialect {

    let name = "automata";
    let cppNamespace = "::mlir::automata";
    let summary = "high performance automata engine";
    let description = [{
        Dialect designed for massive grid simulation, 
        optimized through Loop Tiling and Vectorization.
    }];

}

class Automata_Op<string mnemonic, list<Trait> traits = []> : Op<Automata_Dialect, mnemonic, traits>;  

def CellUpdateOp : Automata_Op<"cell_update", [Pure]> {

    let summary = "Calculate the following state of a cell";
    let description = [{
        Take the current state and its immediate neighbors (von Neumann neighborhood)
        and return the new state based on rules.
    }]; 

    let hasCanonicalizer = 1;

    let arguments = (ins I32:$mid, I32:$left, I32:$up, I32:$right, I32:$down);
    let results = (outs I32:$new_val);
    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` functional-type(operands, results)
    }];
}

def EvolveOp : Automata_Op<"evolve"> {

    let summary = "calculate the next matrix step.";
    let description = [{
        "Take a complete matrix, traverse it, and calculate the next step."
    }];
    let arguments = (ins MemRefOf<[I32]>:$in_array, MemRefOf<[I32]>:$out_array);

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands)
    }];

}