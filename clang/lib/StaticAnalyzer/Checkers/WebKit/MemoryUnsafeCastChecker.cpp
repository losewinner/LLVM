//=======- MemoryUnsafeCastChecker.cpp -------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MemoryUnsafeCast checker, which checks for casts from a
// base type to a derived type.
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class MemoryUnsafeCastChecker : public Checker<check::PreStmt<CastExpr>> {
  BugType BT{this, ""};

public:
  void checkPreStmt(const CastExpr *CE, CheckerContext &C) const;
};
} // end namespace

void emitWarning(CheckerContext &C, const CastExpr &CE, const BugType &BT,
                 QualType FromType, QualType ToType) {
  ExplodedNode *errorNode = C.generateNonFatalErrorNode();
  if (!errorNode)
    return;
  SmallString<192> Buf;
  llvm::raw_svector_ostream OS(Buf);
  OS << "Memory unsafe cast from base type '";
  QualType::print(FromType.getTypePtr(), Qualifiers(), OS, C.getLangOpts(),
                  llvm::Twine());
  OS << "' to derived type '";
  QualType::print(ToType.getTypePtr(), Qualifiers(), OS, C.getLangOpts(),
                  llvm::Twine());
  OS << "'";
  auto R = std::make_unique<PathSensitiveBugReport>(BT, OS.str(), errorNode);
  R->addRange(CE.getSourceRange());
  C.emitReport(std::move(R));
}

void MemoryUnsafeCastChecker::checkPreStmt(const CastExpr *CE,
                                           CheckerContext &C) const {
  auto ExpCast = dyn_cast_or_null<ExplicitCastExpr>(CE);
  if (!ExpCast)
    return;

  auto ToDerivedQualType = ExpCast->getTypeAsWritten();
  auto *SE = CE->getSubExprAsWritten();
  if (ToDerivedQualType->isObjCObjectPointerType()) {
    auto FromBaseQualType = SE->getType();
    bool IsObjCSubType =
        !C.getASTContext().hasSameType(ToDerivedQualType, FromBaseQualType) &&
        C.getASTContext().canAssignObjCInterfaces(
            FromBaseQualType->getAsObjCInterfacePointerType(),
            ToDerivedQualType->getAsObjCInterfacePointerType());
    if (IsObjCSubType)
      emitWarning(C, *CE, BT, FromBaseQualType, ToDerivedQualType);
    return;
  }
  auto ToDerivedType = ToDerivedQualType->getPointeeCXXRecordDecl();
  auto FromBaseType = SE->getType()->getPointeeCXXRecordDecl();
  if (!FromBaseType)
    FromBaseType = SE->getType()->getAsCXXRecordDecl();
  if (!FromBaseType)
    return;
  if (ToDerivedType->isDerivedFrom(FromBaseType))
    emitWarning(C, *CE, BT, SE->getType(), ToDerivedQualType);
}

void ento::registerMemoryUnsafeCastChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<MemoryUnsafeCastChecker>();
}

bool ento::shouldRegisterMemoryUnsafeCastChecker(const CheckerManager &) {
  return true;
}
