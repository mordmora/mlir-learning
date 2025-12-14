/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AutomataDialect.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::automata {

/// Calculate the following state of a cell
/// Take the current state and its immediate neighbors (von Neumann neighborhood)
///         and return the new state based on rules.
class CellUpdateOp;

} // namespace mlir::automata
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::automata {


//===----------------------------------------------------------------------===//
// ::mlir::automata::CellUpdateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CellUpdateOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  CellUpdateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("automata.cell_update", odsAttrs.getContext());
  }

  CellUpdateOpGenericAdaptorBase(CellUpdateOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class CellUpdateOpGenericAdaptor : public detail::CellUpdateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CellUpdateOpGenericAdaptorBase;
  static constexpr int odsIndex_mid = 0;
  static constexpr int odsIndex_left = 1;
  static constexpr int odsIndex_up = 2;
  static constexpr int odsIndex_right = 3;
  static constexpr int odsIndex_down = 4;
public:
  CellUpdateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CellUpdateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CellUpdateOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CellUpdateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CellUpdateOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CellUpdateOpGenericAdaptor(RangeT values, const CellUpdateOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CellUpdateOp, typename = std::enable_if_t<std::is_same_v<LateInst, CellUpdateOp>>>
  CellUpdateOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMid() {
    return (*getODSOperands(0).begin());
  }

  ValueT getLeft() {
    return (*getODSOperands(1).begin());
  }

  ValueT getUp() {
    return (*getODSOperands(2).begin());
  }

  ValueT getRight() {
    return (*getODSOperands(3).begin());
  }

  ValueT getDown() {
    return (*getODSOperands(4).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CellUpdateOpAdaptor : public CellUpdateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CellUpdateOpGenericAdaptor::CellUpdateOpGenericAdaptor;
  CellUpdateOpAdaptor(CellUpdateOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CellUpdateOp : public ::mlir::Op<CellUpdateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<5>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CellUpdateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CellUpdateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_mid = 0;
  static constexpr int odsIndex_left = 1;
  static constexpr int odsIndex_up = 2;
  static constexpr int odsIndex_right = 3;
  static constexpr int odsIndex_down = 4;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("automata.cell_update");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getMid() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getLeft() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getUp() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getRight() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getDown() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(4).begin());
  }

  ::mlir::OpOperand &getMidMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getLeftMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getUpMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRightMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDownMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getNewVal() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type new_val, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down);
  static CellUpdateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type new_val, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down);
  static CellUpdateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type new_val, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down);
  static CellUpdateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down);
  static CellUpdateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value mid, ::mlir::Value left, ::mlir::Value up, ::mlir::Value right, ::mlir::Value down);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CellUpdateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CellUpdateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CellUpdateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CellUpdateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::automata
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::automata::CellUpdateOp)


#endif // GET_OP_CLASSES

