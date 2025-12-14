/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AutomataDialect.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::automata::CellUpdateOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_AutomataDialect1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}
namespace mlir::automata {


//===----------------------------------------------------------------------===//
// ::mlir::automata::CellUpdateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

CellUpdateOpGenericAdaptorBase::CellUpdateOpGenericAdaptorBase(CellUpdateOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
CellUpdateOpAdaptor::CellUpdateOpAdaptor(CellUpdateOp op) : CellUpdateOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CellUpdateOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult CellUpdateOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute CellUpdateOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CellUpdateOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> CellUpdateOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void CellUpdateOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void CellUpdateOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult CellUpdateOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void CellUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type new_val, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down) {
  odsState.addOperands(mid);
  odsState.addOperands(left);
  odsState.addOperands(up);
  odsState.addOperands(right);
  odsState.addOperands(down);
  odsState.addTypes(new_val);
}

CellUpdateOp CellUpdateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type new_val, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(new_val)>(new_val), std::forward<decltype(mid)>(mid), std::forward<decltype(left)>(left), std::forward<decltype(up)>(up), std::forward<decltype(right)>(right), std::forward<decltype(down)>(down));
  auto __res__ = ::llvm::dyn_cast<CellUpdateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

CellUpdateOp CellUpdateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type new_val, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down) {
  return create(builder, builder.getLoc(), std::forward<decltype(new_val)>(new_val), std::forward<decltype(mid)>(mid), std::forward<decltype(left)>(left), std::forward<decltype(up)>(up), std::forward<decltype(right)>(right), std::forward<decltype(down)>(down));
}

void CellUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down) {
  odsState.addOperands(mid);
  odsState.addOperands(left);
  odsState.addOperands(up);
  odsState.addOperands(right);
  odsState.addOperands(down);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CellUpdateOp CellUpdateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(mid)>(mid), std::forward<decltype(left)>(left), std::forward<decltype(up)>(up), std::forward<decltype(right)>(right), std::forward<decltype(down)>(down));
  auto __res__ = ::llvm::dyn_cast<CellUpdateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

CellUpdateOp CellUpdateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(mid)>(mid), std::forward<decltype(left)>(left), std::forward<decltype(up)>(up), std::forward<decltype(right)>(right), std::forward<decltype(down)>(down));
}

void CellUpdateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CellUpdateOp CellUpdateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<CellUpdateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

CellUpdateOp CellUpdateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void CellUpdateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CellUpdateOp CellUpdateOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<CellUpdateOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

CellUpdateOp CellUpdateOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult CellUpdateOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AutomataDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AutomataDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AutomataDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AutomataDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AutomataDialect1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AutomataDialect1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CellUpdateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CellUpdateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  [[maybe_unused]] ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CellUpdateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void CellUpdateOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::automata
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::automata::CellUpdateOp)


#endif // GET_OP_CLASSES

