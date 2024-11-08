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

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class WalkAST : public StmtVisitor<WalkAST> {
  BugReporter &BR;
  const CheckerBase *Checker;
  AnalysisDeclContext* AC;
  ASTContext &ASTC;

public:
  WalkAST(BugReporter &br, const CheckerBase *checker, AnalysisDeclContext *ac)
      : BR(br), Checker(checker), AC(ac), ASTC(AC->getASTContext()) {}

  // Statement visitor methods.
  void VisitChildren(Stmt *S);
  void VisitStmt(Stmt *S) { VisitChildren(S); }
  void VisitCastExpr(CastExpr *CE);
};
} // end anonymous namespace

void emitWarning(QualType FromType, QualType ToType,
                 AnalysisDeclContext *AC, BugReporter &BR,
                 const CheckerBase *Checker,
                 CastExpr *CE) {
  std::string Diagnostics;
  llvm::raw_string_ostream OS(Diagnostics);
  OS << "Unsafe cast from base type '"
     << FromType
     << "' to derived type '"
     << ToType
     << "'",

  BR.EmitBasicReport(
    AC->getDecl(),
    Checker,
    /*Name=*/"Memory unsafe cast",
    categories::SecurityError,
    Diagnostics,
    PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), AC),
    CE->getSourceRange());
}

namespace {
class MemoryUnsafeCastChecker : public Checker<check::ASTCodeBody> {
  BugType BT{this, "Unsafe cast", "WebKit coding guidelines"};
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& Mgr,
                        BugReporter &BR) const {
    WalkAST walker(BR, this, Mgr.getAnalysisDeclContext(D));
    walker.Visit(D->getBody());
  }
};
}

void WalkAST::VisitCastExpr(CastExpr *CE) {
  auto ExpCast = dyn_cast_or_null<ExplicitCastExpr>(CE);
  if (!ExpCast)
    return;

  auto ToDerivedQualType = ExpCast->getTypeAsWritten();
  auto *SE = CE->getSubExprAsWritten();
  if (ToDerivedQualType->isObjCObjectPointerType()) {
    auto FromBaseQualType = SE->getType();
    auto BaseObjCPtrType = FromBaseQualType->getAsObjCInterfacePointerType();
    if (!BaseObjCPtrType)
      return;
    auto DerivedObjCPtrType = ToDerivedQualType->getAsObjCInterfacePointerType();
    if (!DerivedObjCPtrType)
      return;
    bool IsObjCSubType =
        !ASTC.hasSameType(ToDerivedQualType, FromBaseQualType) &&
        ASTC.canAssignObjCInterfaces(BaseObjCPtrType, DerivedObjCPtrType);
    if (IsObjCSubType)
      emitWarning(SE->getType(), ToDerivedQualType, AC, BR, Checker, CE);
    return;
  }
  auto ToDerivedType = ToDerivedQualType->getPointeeCXXRecordDecl();
  if (!ToDerivedType || !ToDerivedType->hasDefinition())
      return;
  auto FromBaseType = SE->getType()->getPointeeCXXRecordDecl();
  if (!FromBaseType)
    FromBaseType = SE->getType()->getAsCXXRecordDecl();
  if (!FromBaseType || !FromBaseType->hasDefinition())
      return;
  if (ToDerivedType->isDerivedFrom(FromBaseType))
    emitWarning(SE->getType(), ToDerivedQualType, AC, BR, Checker, CE);
}

void WalkAST::VisitChildren(Stmt *S) {
  for (Stmt *Child : S->children())
    if (Child)
      Visit(Child);
}

void ento::registerMemoryUnsafeCastChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<MemoryUnsafeCastChecker>();
}

bool ento::shouldRegisterMemoryUnsafeCastChecker(const CheckerManager &mgr) {
  return true;
}
