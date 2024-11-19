//===- CmpPredicate.h - CmpInst Predicate with samesign information -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A CmpInst::Predicate with any samesign information (applicable to ICmpInst).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CMPPREDICATE_H
#define LLVM_IR_CMPPREDICATE_H

#include "llvm/IR/InstrTypes.h"

namespace llvm {
/// An abstraction over a floating-point predicate, and a pack of an integer
/// predicate with samesign information. Functions in ICmpInst construct and
/// return this type in place of a Predicate. It is also implictly constructed
/// with a Predicate, dropping samesign information.
class CmpPredicate {
  CmpInst::Predicate Pred;
  bool HasSameSign;

public:
  CmpPredicate(CmpInst::Predicate Pred, bool HasSameSign = false)
      : Pred(Pred), HasSameSign(HasSameSign) {
    assert(!HasSameSign || CmpInst::isIntPredicate(Pred));
  }

  inline operator CmpInst::Predicate() const { return Pred; }

  inline bool hasSameSign() const { return HasSameSign; }

  static std::optional<CmpPredicate> getMatching(CmpPredicate A,
                                                 CmpPredicate B) {
    if (A.Pred == B.Pred)
      return A.HasSameSign == B.HasSameSign ? A : CmpPredicate(A.Pred);
    if (A.HasSameSign &&
        A.Pred == CmpInst::getFlippedSignednessPredicate(B.Pred))
      return B.Pred;
    if (B.HasSameSign &&
        B.Pred == CmpInst::getFlippedSignednessPredicate(A.Pred))
      return A.Pred;
    return {};
  }

  inline bool operator==(CmpInst::Predicate P) const { return Pred == P; }

  inline bool operator==(CmpPredicate) const = delete;
};
} // namespace llvm

#endif
