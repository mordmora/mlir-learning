/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AutomataDialect.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::automata {

/// high performance automata engine
/// Dialect designed for massive grid simulation, 
///         optimized through Loop Tiling and Vectorization.
class AutomataDialect : public ::mlir::Dialect {
  explicit AutomataDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~AutomataDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("automata");
  }
};

} // namespace mlir::automata
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::automata::AutomataDialect)
