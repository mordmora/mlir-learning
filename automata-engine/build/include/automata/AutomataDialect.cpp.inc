/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AutomataDialect.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::automata::AutomataDialect)
namespace mlir::automata {


AutomataDialect::AutomataDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<AutomataDialect>())
    
     {
  
  initialize();
}

AutomataDialect::~AutomataDialect() = default;


} // namespace mlir::automata
