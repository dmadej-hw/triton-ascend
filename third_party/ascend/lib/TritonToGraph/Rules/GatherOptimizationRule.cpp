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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
// Read-only mirror of TritonToUnstructure::OffsetAnalysis's classification.
//
// GraphOptimizationRule::findCandidates and RewritePlan::revalidate() have no
// IRRewriter (see GraphOptimizationRule.h) -- by design, neither is allowed to
// mutate IR. The real OffsetAnalysis::parse() cannot be used from either: its
// triton::AddPtrOp handler (parseAddPtr) unconditionally inserts a sign-extend
// and an add to materialize a companion offset value, regardless of whether
// anything in the visited subtree is pointer-tracked.
//
// This walker reimplements only the classification rules that actually shape
// the AxisInfo tags GatherOptimization reads (isUnstructured/isStructured/
// isScalarLike/getStructured/getRank); it never reads PtrOffsetInfo::getOffset
// so it never needs the materialized companion value OffsetAnalysis::parse
// builds -- every node kind mirrors its OffsetAnalysis counterpart exactly,
// minus that construction. The one true opacity boundary is triton::LoadOp:
// the real parseLoad always tags a tensor result fully unstructured
// regardless of what its own pointer chain classifies to (only isScalarLike
// would differ, and GatherOptimization never reads isScalarLike on a load
// result), so stopping there means the indices tensor's own address
// computation never needs to be traced. triton::AddPtrOp is deliberately NOT
// an opacity boundary, even though the outer/anchor addptr's pointer operand
// is never classified from analyzeGatherCandidate's own entry point (see
// there): the searches in analyzeGatherCandidate walk both operands of every
// addptr they cross, including ones nested inside the pointer chain (a base
// pointer computed by one addptr, then broadcast and combined with
// per-element offsets by an outer one -- see the non-last-axis example in
// graph-optimize-gather.mlir), so treating every addptr as opaque would
// misclassify a nested one, and anything reading its result, as fully
// unstructured and risk misidentifying it as the indices tensor.
//
// This walker never calls PtrOffsetInfo::setPtr, so .getPtr() reads null on
// every PtrOffsetInfo it produces -- harmless, since nothing here reads it
// either; the base pointer is found independently via
// isSplatOfBlockArgPointer instead (see analyzeGatherCandidate).
class ReadOnlyOffsetClassifier {
public:
  const PtrOffsetInfo &classify(Value value) {
    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;
    // Seed a placeholder before recursing: OffsetAnalysis's own parse() relies
    // on memoization the same way to keep a DAG walk linear, and it also
    // guards against a loop-carried value depending on itself.
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

  // Mirrors OffsetAnalysis::parseBinaryOp, shared by every op it instantiates
  // with (arith::RemSIOp, DivSIOp, the float arithmetic ops, MaxSIOp, MinSIOp,
  // CmpIOp, AndIOp, OrIOp): structured+scalarlike survives only if both
  // operands are scalarLike: otherwise the whole axis range is unstructured.
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

  // Mirrors the tagging shared by both branches of OffsetAnalysis::parseSplat
  // (only a pointer-typed destination additionally materializes a companion
  // offset value, which this walker never needs -- see the call site).
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
      // Mirrors parseAddPtr's classification exactly (combineInfo(ptr,
      // offset)), just without its companion-offset materialization. Not an
      // opacity boundary -- see class comment for why that matters.
      if (auto addPtr = dyn_cast<triton::AddPtrOp>(defOp))
        return combineInfo(classify(addPtr.getPtr()),
                           classify(addPtr.getOffset()));
      // The one true opacity boundary -- see class comment.
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
        // The real parseSplat tags dstStructured (dim==1 -> scalar, else
        // scalarlike) and sets scalarLike=true identically whether or not
        // the element type is a pointer; only a pointer destination also
        // materializes a companion i64 offset splat, which this walker never
        // needs (see class comment). So the same tagging applies here
        // regardless of element type.
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

      // Everything else (arith::SelectOp, FPToSI/SIToFP, triton::ClampFOp,
      // triton::AdvanceOp, ...): mirrors OffsetAnalysis's own dispatch
      // fallback for anything its cascade does not name explicitly.
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
      // Induction variable, or a loop kind whose tied-init we cannot resolve:
      // matches parseLoopRegionIterArg's early return, which leaves the value
      // unclassified and lets parse()'s own fallback mark it unstructured.
      return unstructuredLike(value);
    }

    return unstructuredLike(value);
  }
};

//===----------------------------------------------------------------------===//
// Candidate matching. Every function below is read-only: it may call
// ReadOnlyOffsetClassifier::classify (never OffsetAnalysis::parse) and may
// walk the IR with the read-only Utils.h traversal helpers, but must never
// build or erase an operation.
//===----------------------------------------------------------------------===//

// Marks a tt.load as already handled by this rule: either the generated
// "source"/"fallback" load inside a scf.if it already built, or a load it
// already proved does not match. findCandidates and matchAndRewrite-style
// re-entrancy both rely on this to avoid reconsidering those loads.
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

// Extracts the single splat integer constant multiplied in `mulI`, i.e. the
// literal dimension size a stride-computation MulI encodes.
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

// Finds the literal dimension size that axis `axis` contributes to the
// source tensor's shape: the constant operand of the single arith.muli that
// directly multiplies axis's own tt.expand_dims (its `axis` attribute names
// the physical dimension authoritatively -- no positional bookkeeping is
// needed to attribute it).
//
// This replaces a previous implementation that walked the whole offset
// expression once, front to back, guessing from op *order* how many
// tt.expand_dims a run of arith.muli should cancel out. Anchoring the search
// on each axis's own tt.expand_dims instead means every axis is resolved
// independently, with no cross-axis counter to get out of sync.
//
// The tt.expand_dims for one axis is frequently a shared SSA value with more
// than one arith.muli directly using it -- e.g. the same "logical position"
// tensor gets multiplied by the source tensor's own stride for the pointer
// side of a gather, and separately by a different constant on the branch
// that addresses the indices themselves. Searching *forward* through
// getUsers() for "a" consuming muli would be ambiguous between those. This
// searches *backward* instead, from searchRoot (itself always anchored on
// the pointer side, never inside the indices branch -- see
// analyzeGatherCandidate), for a muli that directly multiplies axis's
// tt.expand_dims. Since the indices-side muli is only reachable *forward*
// from that tt.expand_dims (through its other use), not backward from
// searchRoot, this can never pick the wrong one: the two consumers are
// disambiguated by which one searchRoot's own operand chain leads to, not by
// any property of the tt.expand_dims or muli themselves.
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
  // Never cross into the indices branch or the base-pointer splat: neither
  // can hold this axis's dimension. Do not prune on "a different axis's
  // tt.expand_dims": axes are typically built up by nesting one
  // tt.expand_dims inside another (see e.g. graph-optimize-gather.mlir), so
  // reaching axis 0's muli can require walking through axis 1's node first.
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op);
  };
  // Inclusive of searchRoot itself: it is frequently *is* the axis's own
  // muli (e.g. the outermost/last axis processed).
  Operation *mulI = findOperandDefinitionWithCondition(
      searchRoot->getResult(0), isMulIOfThisAxis, leavesSourceRegion);
  if (!mulI)
    return std::nullopt;
  return extractMulIConstant(cast<arith::MulIOp>(mulI));
}

// Fallback for findAxisDimension when there is no tt.expand_dims to anchor
// on at all: a kernel that processes one row per scf.for iteration (see
// graph-optimize-gather-scalar-row.mlir) can compute axis 1's entire
// absolute position as "(row counter) * axis1_size + local_position",
// splatting the scaled row counter directly instead of building axis 1 up
// with tt.expand_dims/tt.broadcast the way later axes are. In that shape
// there is no tt.expand_dims for findAxisDimension's anchor to find, but the
// dimension is still recoverable: it is simply the constant multiplied by
// the row counter in that one arith.muli.
//
// That same arith.muli's *other* operand -- the row counter itself, still
// unscaled by axis1_size -- is also exactly this rule's rowOffset (see the
// call site). Returning it here, rather than making the caller re-derive it
// with a second, independent search, matters for correctness, not just
// convenience: a generic "first rank-0 value reachable from
// srcAnalysisStart" search (which is what finds rowOffset when this fallback
// is not needed -- see further down) would, in this exact shape, find this
// arith.muli's *scaled* result first, since it sits strictly closer to
// srcAnalysisStart than the unscaled row counter beneath it. Tying rowOffset
// to specifically the operand this axis's own dimension came from is what
// keeps the two consistent with each other.
//
// Only used for axis 1: it is the one axis structurally adjacent to axis 0,
// which is never itself built from tt.expand_dims (its dimension always
// comes directly from the indices' own shape -- see the call site), so it is
// the one place a bare "row counter" can plausibly stand in for what would
// otherwise be a tt.expand_dims-anchored axis.
struct ScalarAxisMatch {
  int64_t dimension;
  Value carrier; // The row counter this axis's dimension scales.
};

std::optional<ScalarAxisMatch> findScalarAxisDimension(Operation *searchRoot,
                                                        Operation *indicesOp) {
  std::function<bool(Operation *)> isMulIWithConstant =
      [](Operation *op) -> bool {
    auto mulI = dyn_cast<arith::MulIOp>(op);
    return mulI && (mulI.getLhs().getDefiningOp<arith::ConstantOp>() ||
                    mulI.getRhs().getDefiningOp<arith::ConstantOp>());
  };
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op);
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

// Read-only counterpart of the former GatherOptimizationConversionPattern::
// analyze(): identical matching rules, but classification comes from
// ReadOnlyOffsetClassifier instead of the mutating OffsetAnalysis::parse, and
// shape recovery is anchored per-axis instead of by a flattened positional
// walk (see findAxisDimension).
std::optional<GatherCandidate> analyzeGatherCandidate(triton::LoadOp loadOp) {
  auto addPtrOp = loadOp.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return std::nullopt;

  auto loadTensorType = dyn_cast<TensorType>(loadOp.getType());
  auto ptrTensorType = dyn_cast<TensorType>(loadOp.getPtr().getType());
  if (!loadTensorType || !ptrTensorType ||
      loadTensorType.getShape() != ptrTensorType.getShape())
    return std::nullopt;

  // Classify only the offset side of the outer addptr, not the pointer side,
  // as an optimization: OffsetAnalysis::parseAddPtr classifies the addptr
  // result as combineInfo(ptrInfo, offsetInfo), the per-axis min of both
  // sides' tags (unstructured < structured < scalarlike < scalar), and a
  // plain tt.splat base pointer always tags scalar/scalarlike (never the
  // smallest), so a bare "ptr = tt.splat %arg" contributes nothing this min
  // could act on -- offsetInfo alone already equals what combining with it
  // would give. This still holds when the pointer side is a longer chain of
  // splat/broadcast/expand_dims (validated against a real 5D example), but is
  // NOT proven in general for a pointer side that itself nests another
  // tt.addptr whose own offset could be unstructured (see
  // graph-optimize-gather.mlir's non-last-axis example, which does have this
  // shape) -- there, this early check could in principle accept an
  // addPtrOp this rule's real combined classification would have rejected.
  // The later checks (isStructuredNotScalar, isSplatOfBlockArgPointer, and
  // per-axis shape recovery all failing to find what they need) are expected
  // to reject any such case regardless, since they inspect the pointer side
  // directly -- but that expectation, unlike this specific min-with-splat
  // argument, is not proven, only exercised by the examples available so far.
  ReadOnlyOffsetClassifier classifier;
  Value offsetValue = addPtrOp.getOffset();
  const PtrOffsetInfo &offsetInfo = classifier.classify(offsetValue);
  if (!offsetInfo.isUnstructured() || offsetInfo.getRank() == 1) {
    // Rank == 1 is the 'index select' pattern, handled elsewhere; only fully
    // unstructured, higher-rank offsets are of interest here.
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

  if (!isIntegerTensorType(candidate.indices.getType(), candidate.indexRank) ||
      candidate.indexRank > 5)
    return std::nullopt;

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

  // Determine the gather axis: the single axis, among all but the leading
  // one, tagged scalarlike in the (pre-index) pointer base's own structure.
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
  if (candidate.gatherAxis == 0)
    return std::nullopt;
  if (candidate.gatherAxis != candidate.indexRank - 1) {
    // Non-last-axis gather is out of scope for this rule.
    return std::nullopt;
  }

  Operation *srcSplat =
      findPrecedingOpWithCondition(analyzedOp, isSplatOfBlockArgPointer, stopIndices);
  if (!srcSplat)
    return std::nullopt;
  candidate.srcPtr = cast<triton::SplatOp>(srcSplat).getSrc();

  ArrayRef<int64_t> indexShape =
      cast<RankedTensorType>(candidate.indices.getType()).getShape();

  // Axis 0 (the row dimension) is taken directly from the indices' own
  // shape, not reconstructed from arithmetic -- it is paired with rowOffset,
  // recovered separately below, exactly as the codegen expects.
  candidate.srcShape.push_back(indexShape[0]);
  std::optional<Value> scalarRowCarrier;
  for (int axis = 1; axis < candidate.indexRank; axis++) {
    if (axis < baseInfo.getRank() && baseStructured[axis] == AxisInfo::scalar) {
      candidate.srcShape.push_back(1);
      continue;
    }
    std::optional<int64_t> dim =
        findAxisDimension(srcAnalysisStart, indicesOp, axis);
    if (!dim && axis == 1) {
      if (std::optional<ScalarAxisMatch> scalarMatch =
              findScalarAxisDimension(srcAnalysisStart, indicesOp)) {
        dim = scalarMatch->dimension;
        scalarRowCarrier = scalarMatch->carrier;
      }
    }
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
    // Axis 1's dimension came from findScalarAxisDimension: its carrier
    // *is* rowOffset (see that function's comment for why this must be tied
    // to the same arith.muli axis 1's dimension came from, rather than
    // re-derived independently below).
    candidate.rowOffset = *scalarRowCarrier;
  } else if (Operation *rank0 = findPrecedingOpWithCondition(
                 srcAnalysisStart,
                 [&](Operation *op) {
                   return classifier.classify(op->getResult(0)).getRank() ==
                          0;
                 },
                 stopIndices)) {
    // Find the first (nearest) scalar (rank-0) value feeding
    // srcAnalysisStart: in a tiled kernel this is normally
    // "program_id * tile_size", the tile's row start, which is exactly what
    // row 0 of the grid built below needs to be offset by.
    //
    // This deliberately does NOT exclude arith::MulIOp results, unlike the
    // pattern this rule replaced. That exclusion looked past a scalar like
    // "program_id * tile_size" to the raw, un-scaled program_id underneath
    // it whenever nothing else (e.g. a loop induction variable added on top,
    // as in graph-optimize-gather.mlir's looped kernel) intervened first --
    // silently using the wrong row offset for any single-shot (non-looped)
    // tile whose row start is a bare multiplication, which is the common
    // case. A rank-0 MulI result is exactly as valid a row offset as any
    // other rank-0 value; nothing here needs it to be one specific op kind.
    // This path is only reachable when scalarRowCarrier above was not set,
    // i.e. axis 1 (if any) was resolved via a real tt.expand_dims, so there
    // is no risk of this landing on an axis-1-scaled intermediate the way an
    // unconditional search from here would in the scalarRowCarrier case.
    candidate.rowOffset = rank0->getResult(0);
  }

  return candidate;
}

// Structural comparison used by revalidate(): a stale plan must not be
// accepted just because *some* gather pattern matches its anchor load again
// -- it must be the *same* pattern it originally found, mirroring how
// DiagonalMaskRemovalPlan/TransposePointwiseReorderPlan compare their
// re-derived candidate against the stored one before trusting it.
bool candidatesMatch(const GatherCandidate &stored, const GatherCandidate &fresh) {
  return stored.loadOp == fresh.loadOp && stored.addPtrOp == fresh.addPtrOp &&
        stored.indices == fresh.indices && stored.srcPtr == fresh.srcPtr &&
        stored.rowOffset == fresh.rowOffset &&
        stored.indexRank == fresh.indexRank &&
        stored.gatherAxis == fresh.gatherAxis &&
        stored.srcShape == fresh.srcShape;
}

//===----------------------------------------------------------------------===//
// reduce<> helper: identical to the former GatherOptimization pass's, moved
// unchanged since it is only ever invoked from apply(), where mutation is
// expected.
//===----------------------------------------------------------------------===//

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
  GatherOptimizationPlan(GatherCandidate candidate, unsigned epoch)
      : loadOp(candidate.loadOp), loadOperation(candidate.loadOp.getOperation()),
        indicesElements(computeIndicesElements(candidate)),
        candidate(std::move(candidate)), epoch(epoch) {}

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
    std::optional<GatherCandidate> fresh = analyzeGatherCandidate(loadOp);
    return success(fresh && candidatesMatch(candidate, *fresh));
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // Re-derive everything from scratch immediately before mutating, exactly
    // like revalidate() -- a stale plan must never be able to mutate the IR.
    std::optional<GatherCandidate> fresh = analyzeGatherCandidate(loadOp);
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

    auto outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, indicesShape, loadTensorType.getElementType());

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
          auto rangeTy = RankedTensorType::get({gatherShape[i]}, i32Type);
          Value range = thenBuilder.create<triton::MakeRangeOp>(
              loc, rangeTy, 0, gatherShape[i]);
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
      auto copyOp = thenBuilder.create<linalg::CopyOp>(
          loc, TypeRange{outputTensor.getType()}, ValueRange{gather},
          ValueRange{outputTensor});
      thenBuilder.create<scf::YieldOp>(loc, copyOp->getResult(0));
    }
    {
      OpBuilder elseBuilder = ifOp.getElseBodyBuilder(rewriter.getListener());
      IRMapping mapping;
      Operation *clonedLoad = elseBuilder.clone(*loadOp, mapping);
      clonedLoad->setAttr(kGatherOptimisedLoadAttr,
                          StringAttr::get(clonedLoad->getContext(), "fallback"));
      auto copyOp = elseBuilder.create<linalg::CopyOp>(
          loc, TypeRange{outputTensor.getType()},
          ValueRange{clonedLoad->getResult(0)}, ValueRange{outputTensor});
      elseBuilder.create<scf::YieldOp>(loc, copyOp->getResult(0));
    }

    rewriter.replaceOp(loadOp, ifOp.getResult(0));
    return success();
  }

  triton::LoadOp loadOp;
  Operation *loadOperation;
  int64_t indicesElements;
  GatherCandidate candidate;
  unsigned epoch;
};

class GatherOptimizationRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::GatherOptimization;
  }

  // The read-only classifier above walks raw IR directly and does not touch
  // any GraphOptimizationContext-owned analysis.
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](triton::LoadOp loadOp) {
      // A load this rule already generated (the "source" or "fallback" load
      // inside a scf.if it already built) must never be reconsidered. Unlike
      // the greedy-pattern-driver version this rule replaces, findCandidates
      // re-walks the whole function from scratch every iteration (see
      // GraphOptimizePass), so a load that simply doesn't match yet needs no
      // "already rejected" marker: it is cheap to -- and safe to -- just
      // re-analyze next time, and doing so lets it start matching again if an
      // earlier rewrite this same pass run changed its shape.
      if (loadOp->getAttr(kGatherOptimisedLoadAttr))
        return;
      std::optional<GatherCandidate> candidate = analyzeGatherCandidate(loadOp);
      if (!candidate)
        return;
      plans.push_back(std::make_unique<GatherOptimizationPlan>(
          std::move(*candidate), context.getEpoch()));
    });
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createGatherOptimizationRule() {
  return std::make_unique<GatherOptimizationRule>();
}
