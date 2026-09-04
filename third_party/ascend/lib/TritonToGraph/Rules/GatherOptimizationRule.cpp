/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToUnstructure/OffsetAnalysis.h"

#include "ascend/include/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "gather-optimization-rule"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

using AxisInfo = PtrOffsetInfo::AxisInfo;

//===----------------------------------------------------------------------===//
// Read-only mirror of OffsetAnalysis's classification (OffsetAnalysis::parse
// mutates IR, so it can't be used here). triton::LoadOp is the one opacity
// boundary; AddPtrOp is not, since callers walk into nested addptrs. .getPtr()
// is always null on the results; the base pointer comes from
// isSplatOfBlockArgPointer instead.
class ReadOnlyOffsetClassifier {
public:
  const PtrOffsetInfo &classify(Value value) {
    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;
    // Seed a placeholder before recursing, in case value is loop-carried.
    PtrOffsetInfo &slot = cache[value];
    slot = classifyUncached(value);
    return cache[value];
  }

private:
  llvm::DenseMap<Value, PtrOffsetInfo> cache;

  static PtrOffsetInfo unstructuredLike(Value value) {
    PtrOffsetInfo info;
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType()))
      info.setUnstructured(tensorType.getRank());
    return info;
  }

  // Mirrors OffsetAnalysis::parseMulI.
  static PtrOffsetInfo combineMulI(const PtrOffsetInfo &lhs,
                                   const PtrOffsetInfo &rhs) {
    PtrOffsetInfo info;
    info.setScalarLike(lhs.isScalarLike() && rhs.isScalarLike());
    auto &lhsStructured = lhs.getStructured();
    auto &rhsStructured = rhs.getStructured();
    size_t maxSize = std::max(lhsStructured.size(), rhsStructured.size());
    auto &dstStructured = info.getStructuredRef();
    dstStructured.resize(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
      if (lhs.isScalarLike())
        dstStructured[i] = rhsStructured[i];
      else if (rhs.isScalarLike())
        dstStructured[i] = lhsStructured[i];
      else
        dstStructured[i] = AxisInfo::unstructured;
    }
    return info;
  }

  // Mirrors OffsetAnalysis::parseBinaryOp.
  static PtrOffsetInfo combineBinaryLike(const PtrOffsetInfo &lhs,
                                         const PtrOffsetInfo &rhs) {
    PtrOffsetInfo info;
    bool scalarLike = lhs.isScalarLike() && rhs.isScalarLike();
    info.setScalarLike(scalarLike);
    int rank = static_cast<int>(lhs.getStructured().size());
    if (scalarLike)
      info.setStructured(rank, AxisInfo::scalarlike);
    else
      info.setUnstructured(rank);
    return info;
  }

  // Mirrors OffsetAnalysis::parseExpandDims (minus the companion offset).
  static PtrOffsetInfo combineExpandDims(const PtrOffsetInfo &src,
                                         unsigned axis) {
    PtrOffsetInfo info;
    info.setScalarLike(src.isScalarLike());
    auto &srcStructured = src.getStructured();
    auto &dstStructured = info.getStructuredRef();
    dstStructured.resize(srcStructured.size() + 1);
    size_t j = 0;
    for (size_t i = 0; i < dstStructured.size(); i++) {
      if (i == axis) {
        dstStructured[i] = AxisInfo::scalar;
      } else {
        dstStructured[i] = srcStructured[j];
        j++;
      }
    }
    return info;
  }

  // Mirrors OffsetAnalysis::parseBroadcast (minus the companion offset).
  static PtrOffsetInfo combineBroadcast(const PtrOffsetInfo &src,
                                        RankedTensorType srcType,
                                        RankedTensorType dstType) {
    PtrOffsetInfo info;
    info.setScalarLike(src.isScalarLike());
    auto broadcastDim = ConverterUtils::getBroadcastDims(srcType, dstType);
    auto &srcStructured = src.getStructured();
    auto &dstStructured = info.getStructuredRef();
    auto dstShape = dstType.getShape();
    dstStructured.resize(srcStructured.size());
    for (size_t i = 0; i < dstStructured.size(); i++) {
      if (llvm::find(broadcastDim, i) != broadcastDim.end() &&
          dstShape[i] != 1)
        dstStructured[i] = AxisInfo::scalarlike;
      else
        dstStructured[i] = srcStructured[i];
    }
    return info;
  }

  // Mirrors OffsetAnalysis::parseSplat's tagging.
  static PtrOffsetInfo splatTagging(RankedTensorType dstType) {
    PtrOffsetInfo info;
    auto &dstStructured = info.getStructuredRef();
    for (auto dim : dstType.getShape())
      dstStructured.push_back(dim == 1 ? AxisInfo::scalar
                                       : AxisInfo::scalarlike);
    info.setScalarLike(true);
    return info;
  }

  // Mirrors OffsetAnalysis::parseConstantOp.
  static PtrOffsetInfo classifyConstant(Value value) {
    PtrOffsetInfo info;
    info.setScalarLike(true);
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
      auto &dstStructured = info.getStructuredRef();
      for (auto dim : tensorType.getShape())
        dstStructured.push_back(dim == 1 ? AxisInfo::scalar
                                         : AxisInfo::scalarlike);
    }
    return info;
  }

  PtrOffsetInfo classifyUncached(Value value) {
    if (Operation *defOp = value.getDefiningOp()) {
      // Mirrors parseAddPtr's classification.
      if (auto addPtr = dyn_cast<triton::AddPtrOp>(defOp))
        return combineInfo(classify(addPtr.getPtr()),
                           classify(addPtr.getOffset()));
      if (isa<triton::LoadOp>(defOp))
        return unstructuredLike(value);

      if (auto addI = dyn_cast<arith::AddIOp>(defOp))
        return combineInfo(classify(addI.getLhs()), classify(addI.getRhs()));
      if (auto subI = dyn_cast<arith::SubIOp>(defOp)) {
        const PtrOffsetInfo &lhs = classify(subI.getLhs());
        const PtrOffsetInfo &rhs = classify(subI.getRhs());
        PtrOffsetInfo info = combineInfo(lhs, rhs);
        if (!(lhs.isStructured() && rhs.isScalarLike()))
          info.setUnstructured(info.getRank());
        return info;
      }
      if (auto mulI = dyn_cast<arith::MulIOp>(defOp))
        return combineMulI(classify(mulI.getLhs()), classify(mulI.getRhs()));
      if (isa<arith::RemSIOp, arith::DivSIOp, arith::MulFOp, arith::DivFOp,
              arith::AddFOp, arith::SubFOp, arith::MinNumFOp,
              arith::MaxNumFOp, arith::MaxSIOp, arith::MinSIOp,
              arith::CmpIOp, arith::AndIOp, arith::OrIOp>(defOp))
        return combineBinaryLike(classify(defOp->getOperand(0)),
                                 classify(defOp->getOperand(1)));
      if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(defOp))
        return combineExpandDims(classify(expandDims.getSrc()),
                                 expandDims.getAxis());
      if (auto broadcast = dyn_cast<triton::BroadcastOp>(defOp)) {
        auto srcType = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
        auto dstType =
            dyn_cast<RankedTensorType>(broadcast.getResult().getType());
        if (!srcType || !dstType)
          return unstructuredLike(value);
        return combineBroadcast(classify(broadcast.getSrc()), srcType,
                                dstType);
      }
      if (auto splat = dyn_cast<triton::SplatOp>(defOp)) {
        if (auto dstType = dyn_cast<RankedTensorType>(splat.getResult().getType()))
          return splatTagging(dstType);
        return unstructuredLike(value);
      }
      if (auto indexCast = dyn_cast<arith::IndexCastOp>(defOp))
        return classify(indexCast.getIn());
      if (auto extSI = dyn_cast<arith::ExtSIOp>(defOp))
        return classify(extSI.getIn());
      if (auto bitcast = dyn_cast<triton::BitcastOp>(defOp))
        return classify(bitcast.getSrc());
      if (isa<arith::ConstantOp, arith::ConstantIntOp, arith::ConstantFloatOp>(
              defOp))
        return classifyConstant(value);
      if (isa<triton::MakeRangeOp>(defOp)) {
        PtrOffsetInfo info;
        info.setStructured(1);
        return info;
      }
      if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(defOp)) {
        PtrOffsetInfo info;
        info.setScalarLike(true);
        return info;
      }

      // Everything else: mirrors OffsetAnalysis's own dispatch fallback.
      return unstructuredLike(value);
    }

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (parentOp && isa<FunctionOpInterface>(parentOp))
        return PtrOffsetInfo(); // matches parse()'s non-pointer argument case
      if (auto loopOp = dyn_cast_or_null<LoopLikeOpInterface>(parentOp)) {
        if (OpOperand *initArgOperand = loopOp.getTiedLoopInit(blockArg))
          return classify(initArgOperand->get());
      }
      // Induction variable, or an unresolvable tied-init: unstructured.
      return unstructuredLike(value);
    }

    return unstructuredLike(value);
  }
};

//===----------------------------------------------------------------------===//
// Candidate matching
//===----------------------------------------------------------------------===//

// Marks a tt.load this rule generated (the source/fallback load in a
// scf.if), so findCandidates skips it.
constexpr llvm::StringLiteral kGatherOptimisedLoadAttr = "gather.optimised.load";

struct GatherCandidate {
  triton::LoadOp loadOp;
  triton::AddPtrOp addPtrOp;
  Value indices;
  Value srcPtr;
  Value rowOffset; // may be null
  int indexRank = 0;
  int gatherAxis = 0;
  SmallVector<int64_t> srcShape;
};

bool isSplatOfBlockArgPointer(Operation *op) {
  auto splatOp = dyn_cast<triton::SplatOp>(op);
  if (!splatOp)
    return false;
  Value src = splatOp.getSrc();
  return isa<BlockArgument>(src) && isa<triton::PointerType>(src.getType());
}

bool isIntegerTensorType(Type type, int &rankOut) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isInteger())
    return false;
  rankOut = tensorType.getShape().size();
  return true;
}

// tt.gather only supports these element types for the gathered value.
bool isSupportedGatherValueType(Type elementType) {
  return elementType.isF16() || elementType.isF32() || elementType.isBF16();
}

std::optional<uint64_t> getByteWidth(Type type) {
  unsigned bitWidth = 0;
  if (auto integerType = dyn_cast<IntegerType>(type))
    bitWidth = integerType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(type))
    bitWidth = floatType.getWidth();
  else
    return std::nullopt;
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;
  return static_cast<uint64_t>(bitWidth / 8);
}

bool checkedMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checkedAddU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

// UB allocations are padded to a 32-byte block on the last axis.
constexpr uint64_t kUbBlockBytes = 32;

uint64_t roundUpToBlock(uint64_t bytes) {
  return (bytes + kUbBlockBytes - 1) / kUbBlockBytes * kUbBlockBytes;
}

// Total bytes for a tensor of this shape/element size, with its last axis
// padded to a 32-byte block (matching how it's actually laid out in UB).
// Returns nullopt on overflow.
std::optional<uint64_t> tileBytes(ArrayRef<int64_t> shape, uint64_t elemBytes) {
  if (shape.empty())
    return elemBytes;
  uint64_t outerElements = 1;
  for (int64_t dim : shape.drop_back())
    if (!checkedMulU64(outerElements, static_cast<uint64_t>(dim),
                       outerElements))
      return std::nullopt;
  uint64_t lastAxisBytes = 0;
  if (!checkedMulU64(static_cast<uint64_t>(shape.back()), elemBytes,
                     lastAxisBytes))
    return std::nullopt;
  uint64_t paddedLastAxisBytes = roundUpToBlock(lastAxisBytes);
  uint64_t total = 0;
  if (!checkedMulU64(outerElements, paddedLastAxisBytes, total))
    return std::nullopt;
  return total;
}

// Empirically fit against real UB-overflow measurements (src/idx shapes of
// varying rank and dtype, both aligned and deliberately unaligned last
// axes): actual usage = 2x the source tile, plus, of the index-shaped
// footprint (last axis padded to a 32-byte block like the source tile),
// 3x at the gather-output element size and 4x at the index element size.
// Exact to the bit across every measurement taken, including mixed
// float16/int32 shapes that separated the two index-footprint terms.
constexpr uint64_t kSrcTileMultiplier = 2;
constexpr uint64_t kIndexFootprintAtLoadElemMultiplier = 3;
constexpr uint64_t kIndexFootprintAtIndexElemMultiplier = 4;

bool exceedsUbCapacity(ArrayRef<int64_t> srcShape,
                       ArrayRef<int64_t> indicesShape, Type loadElementType,
                       Type indexElementType, unsigned ubCapacityBytes) {
  if (ubCapacityBytes == 0)
    return false;
  std::optional<uint64_t> loadElemBytes = getByteWidth(loadElementType);
  std::optional<uint64_t> indexElemBytes = getByteWidth(indexElementType);
  if (!loadElemBytes || !indexElemBytes)
    return true; // Unknown element size: don't risk it.

  std::optional<uint64_t> srcTileBytes = tileBytes(srcShape, *loadElemBytes);
  std::optional<uint64_t> idxAtLoadElemBytes =
      tileBytes(indicesShape, *loadElemBytes);
  std::optional<uint64_t> idxAtIndexElemBytes =
      tileBytes(indicesShape, *indexElemBytes);
  if (!srcTileBytes || !idxAtLoadElemBytes || !idxAtIndexElemBytes)
    return true;
  uint64_t srcContribution = 0, idxContribution1 = 0, idxContribution2 = 0,
           total = 0;
  if (!checkedMulU64(*srcTileBytes, kSrcTileMultiplier, srcContribution) ||
      !checkedMulU64(*idxAtLoadElemBytes, kIndexFootprintAtLoadElemMultiplier,
                     idxContribution1) ||
      !checkedMulU64(*idxAtIndexElemBytes,
                     kIndexFootprintAtIndexElemMultiplier, idxContribution2) ||
      !checkedAddU64(srcContribution, idxContribution1, total) ||
      !checkedAddU64(total, idxContribution2, total))
    return true;

  // ubCapacityBytes is graph_ub_budget_bytes_for_arch (raw/2, a margin sized
  // for StoreCoalescing's less precise model); this formula is fit directly
  // against real overflow measurements, so compare against the raw capacity
  // instead of compounding both margins.
  uint64_t rawUbBytes = 0;
  if (!checkedMulU64(static_cast<uint64_t>(ubCapacityBytes), 2, rawUbBytes))
    return true;

  // A real device crash (VEC "ub address out of bounds") was hit at 196512
  // bytes against a 196608 raw budget -- 96 bytes of "safe" margin per this
  // formula, evidently not actually safe. Reserve 5% of the raw budget
  // (~9.8KB at 192KB) so an unmodeled fixed cost like this can't slip
  // through; this is ~100x the observed overshoot.
  uint64_t safeUbBytes = rawUbBytes - rawUbBytes / 20;
  return total > safeUbBytes;
}

// idx_last threshold below which tt.gather is not expected to beat the
// scalar baseline, indexed by [iteration bucket][src_last bucket]. Derived
// from measured ScalarLoop/tt.gather ratios across rank 2-5 and fp32/fp16
// (see gate_formula_analysis.md): 0 measured fires below ratio 0.98 across
// 1113 fp32 + 26 fp16 validation points.
constexpr int kGatherBenefitThreshold[4][8] = {
    {10, 8, 8, 6, 5, 4, 3, 2}, // T <= 4
    {8, 8, 7, 5, 4, 3, 2, 2},  // 5 <= T <= 16
    {8, 7, 7, 5, 4, 2, 2, 2},  // 17 <= T <= 64
    {8, 7, 7, 4, 3, 2, 2, 2},  // T > 64
};

int gatherSrcBucket(int64_t srcLast) {
  if (srcLast <= 3)
    return 0;
  if (srcLast == 4)
    return 1;
  if (srcLast <= 7)
    return 2;
  if (srcLast <= 9)
    return 3;
  if (srcLast <= 15)
    return 4;
  if (srcLast <= 23)
    return 5;
  if (srcLast <= 31)
    return 6;
  return 7;
}

int gatherIterBucket(int64_t iterations) {
  if (iterations <= 4)
    return 0;
  if (iterations <= 16)
    return 1;
  if (iterations <= 64)
    return 2;
  return 3;
}

// True if rewriting to tt.gather is expected to be at least as fast as the
// scalar baseline it replaces. srcLast/idxLast are the gather axis sizes;
// rowBlk is the rows assigned to one core, rowStep the rows the scalar
// baseline processes per tile iteration (its own tiling, distinct from
// exceedsUbCapacity's tt.gather-side UB budget above).
[[maybe_unused]] bool exceedsBenefitThreshold(int64_t srcLast, int64_t idxLast,
                                              int64_t rowBlk, int64_t rowStep,
                                              uint64_t elemBytes) {
  const int64_t align =
      std::max<int64_t>(1, 32 / static_cast<int64_t>(elemBytes));
  const int64_t iterations = (rowBlk + rowStep - 1) / rowStep;

  if (srcLast == 1) // degenerate source gather axis: its own crossing point
    return idxLast >= 19 ||
          (idxLast >= 2 * align &&
            (idxLast % align == 0 || iterations >= 17 || rowStep == 1));

  if (idxLast % align == 0) // idx tile is a whole number of 32-byte DMA blocks
    return true;
  if (rowStep == 1) // scalar baseline pays full per-iteration overhead/row
    return true;
  return idxLast >= kGatherBenefitThreshold[gatherIterBucket(iterations)]
                                           [gatherSrcBucket(srcLast)];
}

// rowBlk: the upper bound of the nearest enclosing scf.for whose step
// matches rowStep -- the loop tiling the matched pattern's own row
// dimension into rowStep-sized chunks per iteration. Searches from the
// matched op itself, not from rowOffset's defining op: rowOffset is
// typically `program_id * rowBlk`, hoisted outside any such loop. Absent a
// matching loop (or an unresolvable bound), the whole block is handled in
// one shot, i.e. rowBlk == rowStep.
[[maybe_unused]] int64_t findRowBlk(Operation *anchor, int64_t rowStep) {
  auto forOp = anchor->getParentOfType<scf::ForOp>();
  if (!forOp)
    return rowStep;
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  std::optional<int64_t> upperBound =
      getConstantIntValue(forOp.getUpperBound());
  if (!step || *step != rowStep || !upperBound)
    return rowStep;
  return *upperBound;
}

// Extracts the single splat integer constant multiplied in `mulI`.
std::optional<int64_t> extractMulIConstant(arith::MulIOp mulI) {
  arith::ConstantOp constOp = mulI.getRhs().getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    constOp = mulI.getLhs().getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
    if (!denseAttr.getElementType().isInteger() || !denseAttr.isSplat())
      return std::nullopt;
    return denseAttr.getSplatValue<IntegerAttr>().getInt();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getInt();
  return std::nullopt;
}

// Searches backward from searchRoot (anchored on the pointer side), so a
// tt.expand_dims shared with the indices side is never ambiguous.
std::optional<int64_t> findAxisDimension(Operation *searchRoot,
                                         Operation *indicesOp, int64_t axis) {
  auto isAxisExpandDims = [axis](Value v) -> bool {
    auto expandDims = v.getDefiningOp<triton::ExpandDimsOp>();
    return expandDims && static_cast<int64_t>(expandDims.getAxis()) == axis;
  };
  std::function<bool(Operation *)> isMulIOfThisAxis =
      [&](Operation *op) -> bool {
    auto mulI = dyn_cast<arith::MulIOp>(op);
    return mulI && (isAxisExpandDims(mulI.getLhs()) ||
                    isAxisExpandDims(mulI.getRhs()));
  };
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op);
  };
  // Inclusive of searchRoot itself: it is frequently the axis's own muli.
  Operation *mulI = findOperandDefinitionWithCondition(
      searchRoot->getResult(0), isMulIOfThisAxis, leavesSourceRegion);
  if (!mulI)
    return std::nullopt;
  return extractMulIConstant(cast<arith::MulIOp>(mulI));
}

// Last resort: an unmultiplied tt.expand_dims means its multiplier
// canonicalized away as arith.muli(x, 1), so the dimension is 1.
std::optional<int64_t> findFoldedUnitAxisDimension(Operation *searchRoot,
                                                    Operation *indicesOp,
                                                    int64_t axis) {
  std::function<bool(Operation *)> isThisAxisExpandDims =
      [axis](Operation *op) -> bool {
    auto expandDims = dyn_cast<triton::ExpandDimsOp>(op);
    return expandDims && static_cast<int64_t>(expandDims.getAxis()) == axis;
  };
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op);
  };
  if (findOperandDefinitionWithCondition(searchRoot->getResult(0),
                                         isThisAxisExpandDims,
                                         leavesSourceRegion))
    return 1;
  return std::nullopt;
}

// Fallback for axis 1 when there is no tt.expand_dims to anchor on: its
// position can be a plain scalar computation instead of a tensor one (see
// graph-optimize-gather-scalar-row.mlir).
struct ScalarAxisMatch {
  int64_t dimension;
  Value carrier; // Also this rule's rowOffset (see call site).
};

std::optional<ScalarAxisMatch>
findScalarAxisDimension(Operation *searchRoot, Operation *indicesOp,
                        ReadOnlyOffsetClassifier &classifier) {
  // Scalar only, so gatherAxis's own tensor-typed MulI can't shadow this.
  // The non-constant operand must not be scalarLike either: a tile-bound
  // computation like "program_id * row_blk" is scalarLike (pure program_id
  // and constants), while the real per-row counter always incorporates
  // something that genuinely varies per row, which is never scalarLike.
  std::function<bool(Operation *)> isMulIWithConstant =
      [&](Operation *op) -> bool {
    auto mulI = dyn_cast<arith::MulIOp>(op);
    if (!mulI || isa<RankedTensorType>(mulI.getType()))
      return false;
    bool rhsIsConst = static_cast<bool>(
        mulI.getRhs().getDefiningOp<arith::ConstantOp>());
    bool lhsIsConst = static_cast<bool>(
        mulI.getLhs().getDefiningOp<arith::ConstantOp>());
    if (!rhsIsConst && !lhsIsConst)
      return false;
    Value carrier = rhsIsConst ? mulI.getLhs() : mulI.getRhs();
    return !classifier.classify(carrier).isScalarLike();
  };
  // Also stop outside searchRoot's own block (e.g. a hoisted loop-tile bound).
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op) ||
           op->getBlock() != searchRoot->getBlock();
  };
  Operation *mulIOp = findOperandDefinitionWithCondition(
      searchRoot->getResult(0), isMulIWithConstant, leavesSourceRegion);
  if (!mulIOp)
    return std::nullopt;
  auto mulI = cast<arith::MulIOp>(mulIOp);
  std::optional<int64_t> dim = extractMulIConstant(mulI);
  if (!dim)
    return std::nullopt;
  Value carrier = mulI.getRhs().getDefiningOp<arith::ConstantOp>()
                      ? mulI.getLhs()
                      : mulI.getRhs();
  return ScalarAxisMatch{*dim, carrier};
}

// Matches a tt.load against the gather pattern and recovers its shape,
// using ReadOnlyOffsetClassifier rather than the mutating OffsetAnalysis::parse.
std::optional<GatherCandidate> analyzeGatherCandidate(triton::LoadOp loadOp,
                                                       unsigned ubCapacityBytes) {
  auto addPtrOp = loadOp.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return std::nullopt;

  auto loadTensorType = dyn_cast<TensorType>(loadOp.getType());
  auto ptrTensorType = dyn_cast<TensorType>(loadOp.getPtr().getType());
  if (!loadTensorType || !ptrTensorType ||
      loadTensorType.getShape() != ptrTensorType.getShape())
    return std::nullopt;
  if (!isSupportedGatherValueType(loadTensorType.getElementType()))
    return std::nullopt;

  // Classify only the offset side of the outer addptr: a plain tt.splat base
  // pointer never tags below what the offset side gives combineInfo anyway.
  ReadOnlyOffsetClassifier classifier;
  Value offsetValue = addPtrOp.getOffset();
  const PtrOffsetInfo &offsetInfo = classifier.classify(offsetValue);
  if (!offsetInfo.isUnstructured())
    return std::nullopt;
  if (offsetInfo.getRank() == 1) {
    LLVM_DEBUG(llvm::dbgs() << "[GatherOptimization] rank 1 is 'index "
                              "select', handled elsewhere\n");
    return std::nullopt;
  }

  Operation *analyzedOp = addPtrOp.getOperation();

  std::function<bool(Operation *)> isFullyUnstructured =
      [&](Operation *op) -> bool {
    if (auto addIOp = dyn_cast<arith::AddIOp>(op))
      return classifier.classify(addIOp.getLhs()).isUnstructured() &&
             classifier.classify(addIOp.getRhs()).isUnstructured();
    return classifier.classify(op->getResult(0)).isUnstructured();
  };
  std::function<bool(Operation *)> stopStructured = [&](Operation *op) -> bool {
    return classifier.classify(op->getResult(0)).isStructured();
  };
  Operation *indicesOp =
      findPrecedingOpWithCondition(analyzedOp, isFullyUnstructured, stopStructured);
  if (!indicesOp)
    return std::nullopt;

  GatherCandidate candidate;
  candidate.loadOp = loadOp;
  candidate.addPtrOp = addPtrOp;
  candidate.indices = indicesOp->getResult(0);

  if (!isIntegerTensorType(candidate.indices.getType(), candidate.indexRank))
    return std::nullopt;
  if (candidate.indexRank > 5) {
    LLVM_DEBUG(llvm::dbgs() << "[GatherOptimization] rank " << candidate.indexRank
                           << " exceeds the supported maximum of 5\n");
    return std::nullopt;
  }

  std::function<bool(Operation *)> stopIndices = [&](Operation *op) -> bool {
    return op == indicesOp;
  };
  std::function<bool(Operation *)> isStructuredNotScalar =
      [&](Operation *op) -> bool {
    const PtrOffsetInfo &info = classifier.classify(op->getResult(0));
    return info.isStructured() && !info.isScalarLike();
  };
  Operation *srcAnalysisStart = findPrecedingOpWithCondition(
      analyzedOp, isStructuredNotScalar, stopIndices);
  if (!srcAnalysisStart)
    return std::nullopt;

  std::function<bool(Operation *)> isNotUnstructuredNotSplat =
      [&](Operation *op) -> bool {
    return !classifier.classify(op->getResult(0)).isUnstructured() &&
           !isSplatOfBlockArgPointer(op);
  };
  Operation *baseNotUnstructuredOp = findPrecedingOpWithCondition(
      analyzedOp, isNotUnstructuredNotSplat, stopIndices);
  if (!baseNotUnstructuredOp)
    return std::nullopt;

  const PtrOffsetInfo &baseInfo =
      classifier.classify(baseNotUnstructuredOp->getResult(0));
  const SmallVector<AxisInfo> &baseStructured = baseInfo.getStructured();

  // The gather axis is the one tagged scalarlike in the pointer base's own
  // structure: indices vary there, but the base pointer's value doesn't.
  candidate.gatherAxis = 0;
  for (int i = 1; i < baseInfo.getRank(); i++) {
    if (baseStructured[i] == AxisInfo::scalar)
      continue; // 1-element axis: no multiplication, contributes nothing.
    if (baseStructured[i] == AxisInfo::scalarlike) {
      if (candidate.gatherAxis != 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[GatherOptimization] more than one axis could be the "
                      "gather axis\n");
        return std::nullopt;
      }
      candidate.gatherAxis = i;
    }
  }
  if (candidate.gatherAxis == 0) {
    // Size-1 axes tag scalar, not scalarlike, so several can be candidates
    // here with nothing to tell them apart; only accept an unambiguous one.
    SmallVector<int> scalarCandidates;
    for (int i = 1; i < baseInfo.getRank(); i++) {
      if (baseStructured[i] == AxisInfo::scalar)
        scalarCandidates.push_back(i);
    }
    if (scalarCandidates.size() == 1) {
      candidate.gatherAxis = scalarCandidates[0];
    } else if (scalarCandidates.size() > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[GatherOptimization] size-1 gather-axis candidates are "
                    "ambiguous\n");
    }
  }
  if (candidate.gatherAxis == 0)
    return std::nullopt;
  if (candidate.gatherAxis != candidate.indexRank - 1) {
    LLVM_DEBUG(llvm::dbgs() << "[GatherOptimization] detected gather axis "
                           << candidate.gatherAxis
                           << ", but only the last axis is in scope\n");
    return std::nullopt;
  }

  Operation *srcSplat =
      findPrecedingOpWithCondition(analyzedOp, isSplatOfBlockArgPointer, stopIndices);
  if (!srcSplat)
    return std::nullopt;
  candidate.srcPtr = cast<triton::SplatOp>(srcSplat).getSrc();

  ArrayRef<int64_t> indexShape =
      cast<RankedTensorType>(candidate.indices.getType()).getShape();

  // Axis 0 (the row dimension) comes from the indices' own shape, paired
  // with rowOffset, recovered separately below.
  candidate.srcShape.push_back(indexShape[0]);
  std::optional<Value> scalarRowCarrier;
  for (int axis = 1; axis < candidate.indexRank; axis++) {
    std::optional<int64_t> dim =
        findAxisDimension(srcAnalysisStart, indicesOp, axis);
    if (!dim && axis == 1) {
      if (std::optional<ScalarAxisMatch> scalarMatch =
              findScalarAxisDimension(srcAnalysisStart, indicesOp, classifier)) {
        dim = scalarMatch->dimension;
        scalarRowCarrier = scalarMatch->carrier;
      }
    }
    if (!dim)
      dim = findFoldedUnitAxisDimension(srcAnalysisStart, indicesOp, axis);
    if (!dim) {
      LLVM_DEBUG(llvm::dbgs() << "[GatherOptimization] could not recover "
                                 "source dimension for axis "
                              << axis << "\n");
      return std::nullopt;
    }
    candidate.srcShape.push_back(*dim);
  }

  if (candidate.srcShape.size() != indexShape.size())
    return std::nullopt;
  for (size_t d = 0; d < candidate.srcShape.size(); d++) {
    if (static_cast<int>(d) != candidate.gatherAxis &&
        indexShape[d] > candidate.srcShape[d])
      return std::nullopt;
  }

  if (scalarRowCarrier) {
    // Axis 1's dimension came from findScalarAxisDimension: its carrier is
    // rowOffset, tied to the same arith.muli rather than re-derived below.
    candidate.rowOffset = *scalarRowCarrier;
  } else if (Operation *rank0 = findPrecedingOpWithCondition(
                 srcAnalysisStart,
                 [&](Operation *op) {
                   return classifier.classify(op->getResult(0)).getRank() ==
                          0;
                 },
                 stopIndices)) {
    // Nearest scalar (rank-0) value feeding srcAnalysisStart, normally
    // "program_id * tile_size", the tile's row start.
    candidate.rowOffset = rank0->getResult(0);
  }

  if (exceedsUbCapacity(candidate.srcShape, loadTensorType.getShape(),
                       loadTensorType.getElementType(),
                       cast<TensorType>(candidate.indices.getType())
                           .getElementType(),
                       ubCapacityBytes)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[GatherOptimization] estimated UB usage exceeds budget\n");
    return std::nullopt;
  }

  // TEMPORARY, test-only: benefit gate disabled for a full remeasurement
  // pass (gate-conditioned GO data can't show what firing everywhere would
  // actually do -- see gate_formula_fp16_v2.md S2.1). Re-enable before PR.
  //
  // std::optional<uint64_t> loadElemBytes =
  //     getByteWidth(loadTensorType.getElementType());
  // if (!loadElemBytes)
  //   return std::nullopt;
  // int64_t rowStep = candidate.srcShape[0];
  // int64_t rowBlk = findRowBlk(candidate.addPtrOp, rowStep);
  // if (!exceedsBenefitThreshold(candidate.srcShape.back(), indexShape.back(),
  //                              rowBlk, rowStep, *loadElemBytes)) {
  //   LLVM_DEBUG(llvm::dbgs()
  //              << "[GatherOptimization] rewrite not expected to beat the "
  //                 "scalar baseline\n");
  //   return std::nullopt;
  // }

  return candidate;
}

// Used by revalidate(): a stale plan must match the same pattern it
// originally found, not just any gather pattern on its anchor load.
bool candidatesMatch(const GatherCandidate &stored, const GatherCandidate &fresh) {
  return stored.loadOp == fresh.loadOp && stored.addPtrOp == fresh.addPtrOp &&
        stored.indices == fresh.indices && stored.srcPtr == fresh.srcPtr &&
        stored.rowOffset == fresh.rowOffset &&
        stored.indexRank == fresh.indexRank &&
        stored.gatherAxis == fresh.gatherAxis &&
        stored.srcShape == fresh.srcShape;
}

template <typename TIOp>
Value reduce(Value inputTensor, Location loc, IRRewriter &rewriter) {
  auto tensorType = dyn_cast<RankedTensorType>(inputTensor.getType());
  if (!tensorType)
    return nullptr;
  auto elementType = tensorType.getElementType();
  int64_t totalElements = 1;
  for (int64_t dim : tensorType.getShape())
    totalElements *= dim;

  auto flatTensorType = RankedTensorType::get({totalElements}, elementType);
  auto flatTensor =
      rewriter.create<triton::ReshapeOp>(loc, flatTensorType, inputTensor);

  auto reduceOp = rewriter.create<triton::ReduceOp>(
      loc, ValueRange{flatTensor.getResult()}, 0);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<Type, 2> argTypes = {elementType, elementType};
    SmallVector<Location, 2> argLocs = {loc, loc};
    Block *block = rewriter.createBlock(&reduceOp.getRegion(),
                                        reduceOp.getRegion().end(), argTypes,
                                        argLocs);
    auto mathOp = rewriter.create<TIOp>(loc, block->getArgument(0),
                                        block->getArgument(1));
    rewriter.create<triton::ReduceReturnOp>(loc, mathOp.getResult());
  }
  return reduceOp.getResults()[0];
}

//===----------------------------------------------------------------------===//
// Plan / Rule
//===----------------------------------------------------------------------===//

class GatherOptimizationPlan final : public RewritePlan {
public:
  GatherOptimizationPlan(GatherCandidate candidate, unsigned epoch,
                        unsigned ubCapacityBytes)
      : loadOp(candidate.loadOp), loadOperation(candidate.loadOp.getOperation()),
        indicesElements(computeIndicesElements(candidate)),
        candidate(std::move(candidate)), epoch(epoch),
        ubCapacityBytes(ubCapacityBytes) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::GatherOptimization;
  }

  // Larger gathers are worth rewriting first.
  unsigned getBenefit() const override {
    constexpr int64_t maxBenefit = std::numeric_limits<unsigned>::max();
    return static_cast<unsigned>(std::min<int64_t>(indicesElements, maxBenefit));
  }

  Operation *getAnchor() const override { return loadOperation; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getEpoch() != epoch || !loadOp)
      return failure();
    if (loadOp->getParentOfType<triton::FuncOp>().getOperation() !=
        context.getFunction().getOperation())
      return failure();
    if (loadOp->getAttr(kGatherOptimisedLoadAttr))
      return failure();
    std::optional<GatherCandidate> fresh =
        analyzeGatherCandidate(loadOp, ubCapacityBytes);
    return success(fresh && candidatesMatch(candidate, *fresh));
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // Re-derive before mutating, like revalidate(): a stale plan must never
    // be able to mutate the IR.
    std::optional<GatherCandidate> fresh =
        analyzeGatherCandidate(loadOp, ubCapacityBytes);
    if (!fresh || !candidatesMatch(candidate, *fresh))
      return failure();
    return buildGatherRewrite(*fresh, rewriter);
  }

private:
  static int64_t computeIndicesElements(const GatherCandidate &candidate) {
    auto indicesType = dyn_cast<RankedTensorType>(candidate.indices.getType());
    if (!indicesType)
      return 0;
    int64_t elements = 1;
    for (int64_t dim : indicesType.getShape())
      elements *= dim;
    return elements;
  }

  static LogicalResult buildGatherRewrite(const GatherCandidate &candidate,
                                          IRRewriter &rewriter) {
    triton::LoadOp loadOp = candidate.loadOp;
    auto loadTensorType = cast<TensorType>(loadOp.getType());
    auto indicesTensorType = cast<TensorType>(candidate.indices.getType());
    auto indicesShape = indicesTensorType.getShape();
    auto indexType = indicesTensorType.getElementType();
    auto loadElementType = loadTensorType.getElementType();
    ArrayRef<int64_t> srcShape = candidate.srcShape;
    int gatherAxis = candidate.gatherAxis;
    Value rowOffset = candidate.rowOffset;
    Value srcPtr = candidate.srcPtr;
    Value ourIndices = candidate.indices;

    // Both reduce<> calls below share this precondition; check it once here,
    // before either creates anything, rather than relying on them failing
    // identically (true today only because they share the same input).
    if (!isa<RankedTensorType>(ourIndices.getType()))
      return failure();

    Location loc = loadOp.getLoc();
    rewriter.setInsertionPoint(loadOp);

    auto minIndex = reduce<arith::MinSIOp>(ourIndices, loc, rewriter);
    auto maxIndex = reduce<arith::MaxSIOp>(ourIndices, loc, rewriter);
    if (!minIndex || !maxIndex)
      return failure();

    auto minAllowedIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, -srcShape[gatherAxis]));
    auto minCond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, minIndex, minAllowedIndex);
    auto maxAllowedIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, srcShape[gatherAxis]));
    auto maxCond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, maxIndex, maxAllowedIndex);
    auto cond = rewriter.create<arith::AndIOp>(loc, maxCond.getResult(),
                                               minCond.getResult());

    auto ifOp = rewriter.create<scf::IfOp>(loc, /*yieldType=*/loadOp.getType(),
                                           cond, /*hasElse=*/true);
    {
      OpBuilder thenBuilder = ifOp.getThenBodyBuilder(rewriter.getListener());
      SmallVector<int64_t> gatherStrides(srcShape.size()),
          gatherOffsets(srcShape.size()), gatherShape(srcShape.size());
      Value grid = nullptr;
      {
        SmallVector<int64_t> srcStrides(srcShape.size());
        int64_t s = 1;
        for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; i--) {
          srcStrides[i] = s;
          s *= srcShape[i];
        }
        auto i32Type = thenBuilder.getI32Type();
        auto gridTy = RankedTensorType::get(srcShape, i32Type);
        for (size_t i = 0; i < srcShape.size(); i++) {
          gatherOffsets[i] = 0;
          gatherStrides[i] = 1;
          gatherShape[i] =
              (static_cast<int>(i) == gatherAxis) ? srcShape[i] : indicesShape[i];
          // The range here builds the offset grid for loading the full
          // srcShape-sized tile, not the (possibly smaller) gatherShape --
          // that's sliced out of the loaded tile afterwards.
          auto rangeTy = RankedTensorType::get({srcShape[i]}, i32Type);
          Value range = thenBuilder.create<triton::MakeRangeOp>(
              loc, rangeTy, 0, srcShape[i]);
          Value expanded;
          if (i == 0 && rowOffset) {
            auto offset =
                thenBuilder.create<triton::SplatOp>(loc, rangeTy, rowOffset);
            expanded = thenBuilder.create<arith::AddIOp>(loc, range, offset);
          } else {
            expanded = range;
          }
          for (size_t d = 0; d < i; d++) {
            SmallVector<int64_t> expandedShape(
                cast<RankedTensorType>(expanded.getType()).getShape());
            expandedShape.insert(expandedShape.begin(), 1);
            auto expandedTy = RankedTensorType::get(expandedShape, i32Type);
            expanded = thenBuilder.create<triton::ExpandDimsOp>(
                loc, expandedTy, expanded, /*axis=*/0);
          }
          for (size_t d = i + 1; d < srcShape.size(); d++) {
            auto curShape =
                cast<RankedTensorType>(expanded.getType()).getShape();
            SmallVector<int64_t> expandedShape(curShape.begin(), curShape.end());
            expandedShape.push_back(1);
            auto expandedTy = RankedTensorType::get(expandedShape, i32Type);
            expanded = thenBuilder.create<triton::ExpandDimsOp>(
                loc, expandedTy, expanded, /*axis=*/curShape.size());
          }
          Value bcast =
              thenBuilder.create<triton::BroadcastOp>(loc, gridTy, expanded);
          Value strideVal = thenBuilder.create<arith::ConstantIntOp>(
              loc, i32Type, srcStrides[i]);
          Value strideSplat =
              thenBuilder.create<triton::SplatOp>(loc, gridTy, strideVal);
          Value strideMulI =
              thenBuilder.create<arith::MulIOp>(loc, bcast, strideSplat);
          grid = grid ? thenBuilder.create<arith::AddIOp>(loc, grid, strideMulI)
                      : strideMulI;
        }
      }

      // Normalize negative indices: idx += srcShape[axis] & (idx >> (bits-1)).
      auto bitWidth = indexType.getIntOrFloatBitWidth();
      auto cShift = thenBuilder.create<arith::ConstantOp>(
          loc, thenBuilder.getIntegerAttr(indexType, bitWidth - 1));
      auto cShiftTensor =
          thenBuilder.create<triton::SplatOp>(loc, ourIndices.getType(), cShift);
      auto shifted =
          thenBuilder.create<arith::ShRSIOp>(loc, ourIndices, cShiftTensor);
      auto srcColsTensor = thenBuilder.create<triton::SplatOp>(
          loc, indicesTensorType, maxAllowedIndex);
      auto mask = thenBuilder.create<arith::AndIOp>(loc, srcColsTensor, shifted);
      auto indexNormalized =
          thenBuilder.create<arith::AddIOp>(loc, ourIndices, mask);

      auto newSplat = thenBuilder.create<triton::SplatOp>(
          loc, RankedTensorType::get(srcShape, srcPtr.getType()), srcPtr);
      auto addPtr = thenBuilder.create<triton::AddPtrOp>(
          loc, newSplat.getType(), newSplat.getResult(), grid);
      auto load = thenBuilder.create<triton::LoadOp>(
          loc, addPtr, Value(), Value(), loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());
      load->setAttr(kGatherOptimisedLoadAttr,
                    StringAttr::get(load->getContext(), "source"));
      Value input = load.getResult();

      if (gatherShape != ArrayRef<int64_t>(srcShape)) {
        auto resultTy = RankedTensorType::get(gatherShape, loadElementType);
        input = thenBuilder.create<tensor::ExtractSliceOp>(
            loc, resultTy, load, ValueRange{}, ValueRange{}, ValueRange{},
            gatherOffsets, gatherShape, gatherStrides);
      }

      auto gather = thenBuilder.create<triton::GatherOp>(
          loc, loadTensorType, input, indexNormalized,
          candidate.indexRank - 1);
      thenBuilder.create<scf::YieldOp>(loc, gather->getResult(0));
    }
    {
      OpBuilder elseBuilder = ifOp.getElseBodyBuilder(rewriter.getListener());
      IRMapping mapping;
      Operation *clonedLoad = elseBuilder.clone(*loadOp, mapping);
      clonedLoad->setAttr(kGatherOptimisedLoadAttr,
                          StringAttr::get(clonedLoad->getContext(), "fallback"));
      elseBuilder.create<scf::YieldOp>(loc, clonedLoad->getResult(0));
    }

    rewriter.replaceOp(loadOp, ifOp.getResult(0));
    return success();
  }

  triton::LoadOp loadOp;
  Operation *loadOperation;
  int64_t indicesElements;
  GatherCandidate candidate;
  unsigned epoch;
  unsigned ubCapacityBytes;
};

class GatherOptimizationRule final : public GraphOptimizationRule {
public:
  explicit GatherOptimizationRule(unsigned ubCapacityBytes)
      : ubCapacityBytes(ubCapacityBytes) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::GatherOptimization;
  }

  // Walks raw IR directly; no GraphOptimizationContext analysis needed.
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](triton::LoadOp loadOp) {
      // A load that doesn't match yet needs no "rejected" marker: this
      // re-walks the whole function every iteration anyway.
      if (loadOp->getAttr(kGatherOptimisedLoadAttr))
        return;
      std::optional<GatherCandidate> candidate =
          analyzeGatherCandidate(loadOp, ubCapacityBytes);
      if (!candidate)
        return;
      plans.push_back(std::make_unique<GatherOptimizationPlan>(
          std::move(*candidate), context.getEpoch(), ubCapacityBytes));
    });
    return success();
  }

private:
  unsigned ubCapacityBytes;
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createGatherOptimizationRule(unsigned ubCapacityBytes) {
  return std::make_unique<GatherOptimizationRule>(ubCapacityBytes);
}
