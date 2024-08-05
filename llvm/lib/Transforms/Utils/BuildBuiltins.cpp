//===- BuildBuiltins.cpp - Utility builder for builtins -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some functions for lowering compiler builtins.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BuildBuiltins.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

using namespace llvm;

static IntegerType *getIntTy(IRBuilderBase &B, const TargetLibraryInfo *TLI) {
  return B.getIntNTy(TLI->getIntSize());
}

static IntegerType *getSizeTTy(IRBuilderBase &B, const TargetLibraryInfo *TLI) {
  const Module *M = B.GetInsertBlock()->getModule();
  return B.getIntNTy(TLI->getSizeTSize(*M));
}

/// In order to use one of the sized library calls such as
/// __atomic_fetch_add_4, the alignment must be sufficient, the size
/// must be one of the potentially-specialized sizes, and the value
/// type must actually exist in C on the target (otherwise, the
/// function wouldn't actually be defined.)
static bool canUseSizedAtomicCall(unsigned Size, Align Alignment,
                                  const DataLayout &DL) {
  // TODO: "LargestSize" is an approximation for "largest type that
  // you can express in C". It seems to be the case that int128 is
  // supported on all 64-bit platforms, otherwise only up to 64-bit
  // integers are supported. If we get this wrong, then we'll try to
  // call a sized libcall that doesn't actually exist. There should
  // really be some more reliable way in LLVM of determining integer
  // sizes which are valid in the target's C ABI...
  unsigned LargestSize = DL.getLargestLegalIntTypeSizeInBits() >= 64 ? 16 : 8;
  return Alignment >= Size &&
         (Size == 1 || Size == 2 || Size == 4 || Size == 8 || Size == 16) &&
         Size <= LargestSize;
}

Value *llvm::emitAtomicCompareExchangeBuiltin(
    Value *Ptr, Value *ExpectedPtr, Value *DesiredPtr,
    std::variant<Value *, bool> IsWeak, bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> FailureMemorder,
    std::variant<Value *, SyncScope::ID, StringRef> Scope, Value *PrevPtr,
    Type *DataTy, std::optional<uint64_t> DataSize,
    std::optional<uint64_t> AvailableSize, MaybeAlign Align,
    IRBuilderBase &Builder, const DataLayout &DL, const TargetLibraryInfo *TLI,
    const TargetLowering *TL,
    ArrayRef<std::pair<uint32_t, StringRef>> SyncScopes,
    StringRef FallbackScope, bool AllowInstruction, bool AllowSwitch,
    bool AllowSizedLibcall) {
  assert(Ptr->getType()->isPointerTy());
  assert(ExpectedPtr->getType()->isPointerTy());
  assert(DesiredPtr->getType()->isPointerTy());
  assert(TLI);

  LLVMContext &Ctx = Builder.getContext();
  Function *CurFn = Builder.GetInsertBlock()->getParent();

  unsigned MaxAtomicSizeSupported = 16;
  if (TL)
    MaxAtomicSizeSupported = TL->getMaxAtomicSizeInBitsSupported() / 8;

  uint64_t DataSizeConst;
  if (DataSize) {
    DataSizeConst = *DataSize;
  } else {
    TypeSize DS = DL.getTypeStoreSize(DataTy);
    DataSizeConst = DS.getFixedValue();
  }
  uint64_t AvailableSizeConst = AvailableSize.value_or(DataSizeConst);
  assert(DataSizeConst <= AvailableSizeConst);

#ifndef NDEBUG
  if (DataTy) {
    // 'long double' (80-bit extended precision) behaves strange here.
    // DL.getTypeStoreSize says it is 10 bytes
    // Clang says it is 12 bytes
    // AtomicExpandPass would disagree with CGAtomic (not for cmpxchg that does
    // not support floats, so AtomicExpandPass doesn't even know it originally
    // was an FP80)
    TypeSize DS = DL.getTypeStoreSize(DataTy);
    assert(DS.getKnownMinValue() <= DataSizeConst &&
           "Must access at least all the relevant bits of the data, possibly "
           "some more for padding");
  }
#endif

  Type *BoolTy = Builder.getInt1Ty();
  Type *IntTy = getIntTy(Builder, TLI);

  uint64_t PreferredSize = PowerOf2Ceil(DataSizeConst);
  if (!PreferredSize || PreferredSize > MaxAtomicSizeSupported)
    PreferredSize = DataSizeConst;

  llvm::Align EffectiveAlign;
  if (Align) {
    EffectiveAlign = *Align;
  } else {
    // https://llvm.org/docs/LangRef.html#cmpxchg-instruction
    //
    //   The alignment is only optional when parsing textual IR; for in-memory
    //   IR, it is always present. If unspecified, the alignment is assumed to
    //   be equal to the size of the ‘<value>’ type.
    //
    // We prefer safety here and assume no alignment, unless
    // getPointerAlignment() can determine the actual alignment.
    EffectiveAlign = Ptr->getPointerAlignment(DL);
  }

  // Only use the original data type if it is compatible with cmpxchg (and sized
  // libcall function) and matches the preferred size. No type punning needed
  // for __atomic_compare_exchange which only takes pointers.
  Type *CoercedTy = nullptr;
  if (DataTy && DataSizeConst == PreferredSize &&
      (DataTy->isIntegerTy() || DataTy->isPointerTy()))
    CoercedTy = DataTy;
  else if (PreferredSize <= 16)
    CoercedTy = IntegerType::get(Ctx, PreferredSize * 8);

  // For resolving the SuccessMemorder/FailureMemorder arguments. If it is
  // constant, determine the AtomicOrdering for use with the cmpxchg
  // instruction. Also determines the llvm::Value to be passed to
  // __atomic_compare_exchange in case cmpxchg is not legal.
  auto processMemorder = [&](auto MemorderVariant)
      -> std::pair<std::optional<AtomicOrdering>, Value *> {
    if (std::holds_alternative<AtomicOrdering>(MemorderVariant)) {
      auto Memorder = std::get<AtomicOrdering>(MemorderVariant);
      return std::make_pair(
          Memorder,
          ConstantInt::get(IntTy, static_cast<uint64_t>(toCABI(Memorder))));
    }
    if (std::holds_alternative<AtomicOrderingCABI>(MemorderVariant)) {
      auto MemorderCABI = std::get<AtomicOrderingCABI>(MemorderVariant);
      return std::make_pair(
          fromCABI(MemorderCABI),
          ConstantInt::get(IntTy, static_cast<uint64_t>(MemorderCABI)));
    }

    auto *MemorderCABI = std::get<Value *>(MemorderVariant);
    if (auto *MO = dyn_cast<ConstantInt>(MemorderCABI)) {
      uint64_t MOInt = MO->getZExtValue();
      return std::make_pair(fromCABI(MOInt), MO);
    }

    return std::make_pair(std::nullopt, MemorderCABI);
  };

  auto processIsWeak =
      [&](auto WeakVariant) -> std::pair<std::optional<bool>, Value *> {
    if (std::holds_alternative<bool>(WeakVariant)) {
      bool IsWeakBool = std::get<bool>(WeakVariant);
      return std::make_pair(IsWeakBool, Builder.getInt1(IsWeakBool));
    }

    auto *BoolVal = std::get<Value *>(WeakVariant);
    if (auto *BoolConst = dyn_cast<ConstantInt>(BoolVal)) {
      uint64_t IsWeakBool = BoolConst->getZExtValue();
      return std::make_pair(IsWeakBool != 0, BoolVal);
    }

    return std::make_pair(std::nullopt, BoolVal);
  };

  auto processScope = [&](auto ScopeVariant)
      -> std::pair<std::optional<SyncScope::ID>, Value *> {
    if (std::holds_alternative<SyncScope::ID>(ScopeVariant)) {
      auto ScopeID = std::get<SyncScope::ID>(ScopeVariant);
      return std::make_pair(ScopeID, nullptr);
    }

    if (std::holds_alternative<StringRef>(ScopeVariant)) {
      auto ScopeName = std::get<StringRef>(ScopeVariant);
      SyncScope::ID ScopeID = Ctx.getOrInsertSyncScopeID(ScopeName);
      return std::make_pair(ScopeID, nullptr);
    }

    auto *IntVal = std::get<Value *>(ScopeVariant);
    if (auto *InstConst = dyn_cast<ConstantInt>(IntVal)) {
      uint64_t ScopeVal = InstConst->getZExtValue();
      return std::make_pair(ScopeVal, IntVal);
    }

    return std::make_pair(std::nullopt, IntVal);
  };

  auto [IsWeakConst, IsWeakVal] = processIsWeak(IsWeak);
  auto [SuccessMemorderConst, SuccessMemorderCABI] =
      processMemorder(SuccessMemorder);
  auto [FailureMemorderConst, FailureMemorderCABI] =
      processMemorder(FailureMemorder);
  auto [ScopeConst, ScopeVal] = processScope(Scope);

  // Fix malformed inputs. We do not want to emit illegal IR.
  //
  // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
  //
  //   [failure_memorder] This memory order cannot be __ATOMIC_RELEASE nor
  //   __ATOMIC_ACQ_REL. It also cannot be a stronger order than that
  //   specified by success_memorder.
  //
  // https://llvm.org/docs/LangRef.html#cmpxchg-instruction
  //
  //   Both ordering parameters must be at least monotonic, the failure
  //   ordering cannot be either release or acq_rel.
  //
  if (FailureMemorderConst &&
      ((*FailureMemorderConst == AtomicOrdering::Release) ||
       (*FailureMemorderConst == AtomicOrdering::AcquireRelease))) {
    // Fall back to monotonic atomic when illegal value is passed. As with the
    // dynamic case below, it is an arbitrary choice.
    FailureMemorderConst = AtomicOrdering::Monotonic;
  }
  if (FailureMemorderConst && SuccessMemorderConst &&
      !isAtLeastOrStrongerThan(*SuccessMemorderConst, *FailureMemorderConst)) {
    // Make SuccessMemorder as least as strong as FailureMemorder
    SuccessMemorderConst =
        getMergedAtomicOrdering(*SuccessMemorderConst, *FailureMemorderConst);
  }

  // https://llvm.org/docs/LangRef.html#cmpxchg-instruction
  //
  //   The type of ‘<cmp>’ must be an integer or pointer type whose bit width is
  //   a power of two greater than or equal to eight and less than or equal to a
  //   target-specific size limit.
  bool CanUseCmpxchngInst = PreferredSize <= MaxAtomicSizeSupported &&
                            llvm::isPowerOf2_64(PreferredSize) && CoercedTy;
  bool CanUseSingleCmpxchngInst = CanUseCmpxchngInst && SuccessMemorderConst &&
                                  FailureMemorderConst && IsWeakConst &&
                                  ScopeConst;
  bool CanUseSizedLibcall =
      canUseSizedAtomicCall(PreferredSize, EffectiveAlign, DL);

  Value *ExpectedVal;
  Value *DesiredVal;

  // Emit cmpxchg instruction, either as a single instruction, or as a case of a
  // per-constant switch.
  auto EmitCmpxchngInst = [&](bool IsWeak, SyncScope::ID Scope,
                              AtomicOrdering SuccessMemorder,
                              AtomicOrdering FailureMemorder) {
    AtomicCmpXchgInst *AtomicInst =
        Builder.CreateAtomicCmpXchg(Ptr, ExpectedVal, DesiredVal, Align,
                                    SuccessMemorder, FailureMemorder, Scope);
    AtomicInst->setName("cmpxchg.pair");
    AtomicInst->setAlignment(EffectiveAlign);
    AtomicInst->setWeak(IsWeak);
    AtomicInst->setVolatile(IsVolatile);

    if (PrevPtr) {
      Value *PreviousVal =
          Builder.CreateExtractValue(AtomicInst, /*Idxs=*/0, "cmpxchg.prev");
      Builder.CreateStore(PreviousVal, PrevPtr);
    }

    Value *SuccessFailureVal =
        Builder.CreateExtractValue(AtomicInst, /*Idxs=*/1, "cmpxchg.success");

    assert(SuccessFailureVal->getType()->isIntegerTy(1));
    return SuccessFailureVal;
  };

  if (CanUseSingleCmpxchngInst && AllowInstruction) {
    // FIXME: Need to get alignment correct
    ExpectedVal =
        Builder.CreateLoad(CoercedTy, ExpectedPtr, "cmpxchg.expected");
    DesiredVal = Builder.CreateLoad(CoercedTy, DesiredPtr, "cmpxchg.desired");
    return EmitCmpxchngInst(*IsWeakConst, *ScopeConst, *SuccessMemorderConst,
                            *FailureMemorderConst);
  }

  // Switching only needed for cmpxchg instruction which requires constant
  // arguments.
  // FIXME: If AtomicExpandPass later considers the cmpxchg not lowerable for
  // the given target, it will also generate a call to the
  // __atomic_compare_exchange function. In that case the switching was very
  // unnecessary but cannot be undone.
  if (CanUseCmpxchngInst && AllowSwitch && AllowInstruction) {
    auto createBasicBlock = [&](const Twine &Name) {
      return BasicBlock::Create(Ctx, Name, CurFn);
    };

    ExpectedVal =
        Builder.CreateLoad(CoercedTy, ExpectedPtr, "cmpxchg.expected");
    DesiredVal = Builder.CreateLoad(CoercedTy, DesiredPtr, "cmpxchg.desired");

    auto GenFailureMemorderSwitch =
        [&](bool IsWeak, SyncScope::ID Scope,
            AtomicOrdering SuccessMemorder) -> Value * {
      if (FailureMemorderConst) {
        // FIXME:  (from CGAtomic)
        // 31.7.2.18: "The failure argument shall not be memory_order_release
        // nor memory_order_acq_rel". Fallback to monotonic.
        //
        // Prior to c++17, "the failure argument shall be no stronger than the
        // success argument". This condition has been lifted and the only
        // precondition is 31.7.2.18. Effectively treat this as a DR and skip
        // language version checks.
        return EmitCmpxchngInst(IsWeak, Scope, SuccessMemorder,
                                *FailureMemorderConst);
      }

      // Create all the relevant BB's
      BasicBlock *MonotonicBB = createBasicBlock("monotonic_fail");
      BasicBlock *AcquireBB = createBasicBlock("acquire_fail");
      BasicBlock *SeqCstBB = createBasicBlock("seqcst_fail");
      BasicBlock *ContBB = createBasicBlock("atomic.continue");

      // MonotonicBB is arbitrarily chosen as the default case; in practice,
      // this doesn't matter unless someone is crazy enough to use something
      // that doesn't fold to a constant for the ordering.
      llvm::SwitchInst *SI =
          Builder.CreateSwitch(FailureMemorderCABI, MonotonicBB);
      // Implemented as acquire, since it's the closest in LLVM.
      SI->addCase(
          Builder.getInt32(static_cast<int32_t>(AtomicOrderingCABI::consume)),
          AcquireBB);
      SI->addCase(
          Builder.getInt32(static_cast<int32_t>(AtomicOrderingCABI::acquire)),
          AcquireBB);
      SI->addCase(
          Builder.getInt32(static_cast<int32_t>(AtomicOrderingCABI::seq_cst)),
          SeqCstBB);

      // Emit all the different atomics
      Builder.SetInsertPoint(MonotonicBB);
      Value *MonotonicResult = EmitCmpxchngInst(IsWeak, Scope, SuccessMemorder,
                                                AtomicOrdering::Monotonic);
      BasicBlock *MonotonicSourceBB = Builder.GetInsertBlock();
      Builder.CreateBr(ContBB);

      Builder.SetInsertPoint(AcquireBB);
      Value *AcquireResult = EmitCmpxchngInst(IsWeak, Scope, SuccessMemorder,
                                              AtomicOrdering::Acquire);
      BasicBlock *AcquireSourceBB = Builder.GetInsertBlock();
      Builder.CreateBr(ContBB);

      Builder.SetInsertPoint(SeqCstBB);
      Value *SeqCstResult =
          EmitCmpxchngInst(IsWeak, Scope, SuccessMemorder,
                           AtomicOrdering::SequentiallyConsistent);
      BasicBlock *SeqCstSourceBB = Builder.GetInsertBlock();
      Builder.CreateBr(ContBB);

      Builder.SetInsertPoint(ContBB);
      PHINode *Result = Builder.CreatePHI(BoolTy, 3, "cmpxcgh.success");
      Result->addIncoming(MonotonicResult, MonotonicSourceBB);
      Result->addIncoming(AcquireResult, AcquireSourceBB);
      Result->addIncoming(SeqCstResult, SeqCstSourceBB);
      return Result;
    };

    auto GenSuccessMemorderSwitch = [&](bool IsWeak,
                                        SyncScope::ID Scope) -> Value * {
      if (SuccessMemorderConst)
        return GenFailureMemorderSwitch(IsWeak, Scope, *SuccessMemorderConst);

      // Create all the relevant BB's
      BasicBlock *MonotonicBB = createBasicBlock("monotonic");
      BasicBlock *AcquireBB = createBasicBlock("acquire");
      BasicBlock *ReleaseBB = createBasicBlock("release");
      BasicBlock *AcqRelBB = createBasicBlock("acqrel");
      BasicBlock *SeqCstBB = createBasicBlock("seqcst");
      BasicBlock *ContBB = createBasicBlock("atomic.continue");

      // Create the switch for the split
      // MonotonicBB is arbitrarily chosen as the default case; in practice,
      // this doesn't matter unless someone is crazy enough to use something
      // that doesn't fold to a constant for the ordering.
      Value *Order = Builder.CreateIntCast(SuccessMemorderCABI,
                                           Builder.getInt32Ty(), false);
      llvm::SwitchInst *SI = Builder.CreateSwitch(Order, MonotonicBB);

      Builder.SetInsertPoint(ContBB);
      PHINode *Result = Builder.CreatePHI(BoolTy, 5, "cmpxcgh.success");

      // Emit all the different atomics
      Builder.SetInsertPoint(MonotonicBB);
      Value *MonotonicResult =
          GenFailureMemorderSwitch(IsWeak, Scope, AtomicOrdering::Monotonic);
      Result->addIncoming(MonotonicResult, MonotonicBB);
      Builder.CreateBr(ContBB);

      Builder.SetInsertPoint(AcquireBB);
      Value *AcquireResult =
          GenFailureMemorderSwitch(IsWeak, Scope, AtomicOrdering::Acquire);
      Builder.CreateBr(ContBB);
      SI->addCase(
          Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::consume)),
          Builder.GetInsertBlock());
      SI->addCase(
          Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::acquire)),
          Builder.GetInsertBlock());
      Result->addIncoming(AcquireResult, AcquireBB);

      Builder.SetInsertPoint(ReleaseBB);
      Value *ReleaseResult =
          GenFailureMemorderSwitch(IsWeak, Scope, AtomicOrdering::Release);
      Builder.CreateBr(ContBB);
      SI->addCase(
          Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::release)),
          Builder.GetInsertBlock());
      Result->addIncoming(ReleaseResult, Builder.GetInsertBlock());

      Builder.SetInsertPoint(AcqRelBB);
      Value *AcqRelResult = GenFailureMemorderSwitch(
          IsWeak, Scope, AtomicOrdering::AcquireRelease);
      Builder.CreateBr(ContBB);
      SI->addCase(
          Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::acq_rel)),
          AcqRelBB);
      Result->addIncoming(AcqRelResult, Builder.GetInsertBlock());

      Builder.SetInsertPoint(SeqCstBB);
      Value *SeqCstResult = GenFailureMemorderSwitch(
          IsWeak, Scope, AtomicOrdering::SequentiallyConsistent);
      Builder.CreateBr(ContBB);
      SI->addCase(
          Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::seq_cst)),
          SeqCstBB);
      Result->addIncoming(SeqCstResult, Builder.GetInsertBlock());

      Builder.SetInsertPoint(Result->getNextNode());
      return Result;
    };

    auto GenScopeSwitch = [&](bool IsWeak) -> Value * {
      if (ScopeConst)
        return GenSuccessMemorderSwitch(IsWeak, *ScopeConst);

      // Handle non-constant scope.
      DenseMap<unsigned, BasicBlock *> BB;
      for (const auto &S : SyncScopes) {
        if (FallbackScope == S.second)
          continue; // always the default case
        BB[S.first] = createBasicBlock(Twine("cmpxchg.scope.") + S.second);
      }

      BasicBlock *DefaultBB = createBasicBlock("atomic.scope.fallback");
      BasicBlock *ContBB = createBasicBlock("atomic.scope.continue");

      Builder.SetInsertPoint(ContBB);
      PHINode *Result =
          Builder.CreatePHI(BoolTy, SyncScopes.size() + 1, "cmpxchg.success");

      Value *SC = Builder.CreateIntCast(ScopeVal, Builder.getInt32Ty(),
                                        /*IsSigned*/ false,
                                        "atomic.cmpxchg.scope.cast");
      // If unsupported synch scope is encountered at run time, assume a
      // fallback synch scope value.
      SwitchInst *SI = Builder.CreateSwitch(SC, DefaultBB);
      for (const auto &S : SyncScopes) {
        BasicBlock *B = BB[S.first];
        SI->addCase(Builder.getInt32(S.first), B);

        Builder.SetInsertPoint(B);
        SyncScope::ID SyncScopeID = Ctx.getOrInsertSyncScopeID(S.second);
        Value *SyncResult = GenSuccessMemorderSwitch(IsWeak, SyncScopeID);
        Result->addIncoming(SyncResult, Builder.GetInsertBlock());
        Builder.CreateBr(ContBB);
      }

      Builder.SetInsertPoint(DefaultBB);
      SyncScope::ID SyncScopeID = Ctx.getOrInsertSyncScopeID(FallbackScope);
      Value *DefaultResult = GenSuccessMemorderSwitch(IsWeak, SyncScopeID);
      Result->addIncoming(DefaultResult, Builder.GetInsertBlock());
      Builder.CreateBr(ContBB);

      Builder.SetInsertPoint(Result->getNextNode());
      return Result;
    };

    auto GenWeakSwitch = [&]() -> Value * {
      if (IsWeakConst)
        return GenScopeSwitch(*IsWeakConst);

      // Create all the relevant BB's
      BasicBlock *StrongBB = createBasicBlock("cmpxchg.strong");
      BasicBlock *WeakBB = createBasicBlock("cmpxchg.weak");
      BasicBlock *ContBB = createBasicBlock("cmpxchg.continue");

      // FIXME: Why is this a switch?
      llvm::SwitchInst *SI = Builder.CreateSwitch(IsWeakVal, WeakBB);
      SI->addCase(Builder.getInt1(false), StrongBB);

      Builder.SetInsertPoint(StrongBB);
      Value *StrongResult = GenScopeSwitch(false);
      BasicBlock *StrongSourceBB = Builder.GetInsertBlock();
      Builder.CreateBr(ContBB);

      Builder.SetInsertPoint(WeakBB);
      Value *WeakResult = GenScopeSwitch(true);
      BasicBlock *WeakSourceBB = Builder.GetInsertBlock();
      Builder.CreateBr(ContBB);

      Builder.SetInsertPoint(ContBB);
      PHINode *Result = Builder.CreatePHI(BoolTy, 2, "cmpxchg.isweak.success");
      Result->addIncoming(WeakResult, WeakSourceBB);
      Result->addIncoming(StrongResult, StrongSourceBB);
      return Result;
    };

    return GenWeakSwitch();
  }

  // Fallback to a libcall function. From here on IsWeak/Scope/IsVolatile is
  // ignored. IsWeak is assumed to be false, Scope is assumed to be
  // SyncScope::System (strongest possible assumption synchronizing with
  // everything, instead of just a subset of sibling threads), and volatile does
  // not apply to function calls.

  // FIXME: Some AMDGCN regression tests the addrspace, but
  // __atomic_compare_exchange by definition is addrsspace(0) and
  // emitAtomicCompareExchange will complain about it.
  if (Ptr->getType()->getPointerAddressSpace() ||
      ExpectedPtr->getType()->getPointerAddressSpace() ||
      DesiredPtr->getType()->getPointerAddressSpace())
    return Builder.getInt1(false);

  if (CanUseSizedLibcall && AllowSizedLibcall) {
    LoadInst *DesiredVal =
        Builder.CreateLoad(IntegerType::get(Ctx, PreferredSize * 8), DesiredPtr,
                           "cmpxchg.desired");
    Value *SuccessResult = emitAtomicCompareExchangeN(
        PreferredSize, Ptr, ExpectedPtr, DesiredVal, SuccessMemorderCABI,
        FailureMemorderCABI, Builder, DL, TLI);
    if (SuccessResult) {
      Value *SuccessBool =
          Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, SuccessResult,
                            Builder.getInt8(0), "cmpxchg.success");

      if (PrevPtr && PrevPtr != ExpectedPtr)
        Builder.CreateMemCpy(PrevPtr, {}, ExpectedPtr, {}, DataSizeConst);
      return SuccessBool;
    }

    // emitAtomicCompareExchangeN can return nullptr if the backend does not
    // support sized libcalls. Fall back to the non-sized libcall and remove the
    // unused load again.
    DesiredVal->eraseFromParent();
  }

  // FIXME: emitAtomicCompareExchange may fail if a function declaration with
  // the same name but different signature has already been emitted. Since the
  // function name starts with "__", i.e. is reserved for use by the compiler,
  // this should not happen.
  // It may also fail if the target's TargetLibraryInfo claims that
  // __atomic_compare_exchange is not supported. In either case there is no
  // fallback for atomics not supported by the target and we have to crash. 
  Value *SuccessResult = emitAtomicCompareExchange(
      ConstantInt::get(getSizeTTy(Builder, TLI), DataSizeConst), Ptr,
      ExpectedPtr, DesiredPtr, SuccessMemorderCABI, FailureMemorderCABI,
      Builder, DL, TLI);
  if (!SuccessResult)
    report_fatal_error("expandAtomicOpToLibcall shouldn't fail for CAS");

  Value *SuccessBool =
      Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, SuccessResult,
                        Builder.getInt8(0), "cmpxchg.success");

  if (PrevPtr && PrevPtr != ExpectedPtr)
    Builder.CreateMemCpy(PrevPtr, {}, ExpectedPtr, {}, DataSizeConst);
  return SuccessBool;
}

Value *llvm::emitAtomicCompareExchangeBuiltin(
    Value *Ptr, Value *ExpectedPtr, Value *DesiredPtr,
    std::variant<Value *, bool> Weak, bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> FailureMemorder,
    Value *PrevPtr, Type *DataTy, std::optional<uint64_t> DataSize,
    std::optional<uint64_t> AvailableSize, MaybeAlign Align,
    IRBuilderBase &Builder, const DataLayout &DL, const TargetLibraryInfo *TLI,
    const TargetLowering *TL, bool AllowInstruction, bool AllowSwitch,
    bool AllowSizedLibcall) {
  return emitAtomicCompareExchangeBuiltin(
      Ptr, ExpectedPtr, DesiredPtr, Weak, IsVolatile, SuccessMemorder,
      FailureMemorder, SyncScope::System, PrevPtr, DataTy, DataSize,
      AvailableSize, Align, Builder, DL, TLI, TL, {}, StringRef(),
      AllowInstruction, AllowSwitch, AllowSizedLibcall);
}
