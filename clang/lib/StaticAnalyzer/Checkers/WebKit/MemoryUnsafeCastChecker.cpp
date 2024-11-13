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
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"

using namespace clang;
using namespace ento;
using namespace ast_matchers;

namespace {
static constexpr const char *const BaseNode = "BaseNode";
static constexpr const char *const DerivedNode = "DerivedNode";
static constexpr const char *const WarnRecordDecl = "WarnRecordDecl";

class MemoryUnsafeCastChecker : public Checker<check::ASTCodeBody> {
  BugType BT{this, "Unsafe cast", "WebKit coding guidelines"};
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& Mgr,
                        BugReporter &BR) const;
};
}  // end namespace

static void emitDiagnostics(const BoundNodes &Nodes, BugReporter &BR,
                            AnalysisDeclContext *ADC,
                            const MemoryUnsafeCastChecker *Checker) {
  const auto *CE = Nodes.getNodeAs<CastExpr>(WarnRecordDecl);
  const NamedDecl *Base = Nodes.getNodeAs<NamedDecl>(BaseNode);
  const NamedDecl *Derived = Nodes.getNodeAs<NamedDecl>(DerivedNode);
  assert(CE && Base && Derived);

  std::string Diagnostics;
  llvm::raw_string_ostream OS(Diagnostics);
  OS << "Unsafe cast from base type '" << Base->getNameAsString()
     << "' to derived type '" << Derived->getNameAsString() << "'";

  BR.EmitBasicReport(
      ADC->getDecl(), Checker,
      /*Name=*/"OSObject C-Style Cast", categories::SecurityError,
      Diagnostics,
      PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), ADC),
      CE->getSourceRange());
}

namespace clang {
namespace ast_matchers {
AST_MATCHER_P(StringLiteral, mentionsBoundType, std::string, BindingID) {
  return Builder->removeBindings([this, &Node](const BoundNodesMap &Nodes) {
    const auto &BN = Nodes.getNode(this->BindingID);
    if (const auto *ND = BN.get<NamedDecl>()) {
      return ND->getName() != Node.getString();
    }
    return true;
  });
}
} // end namespace ast_matchers
} // end namespace clang

static decltype(auto) hasTypePointingTo(DeclarationMatcher DeclM) {
  return hasType(pointerType(pointee(hasDeclaration(DeclM))));
}

void MemoryUnsafeCastChecker::checkASTCodeBody(const Decl *D,
                                               AnalysisManager &AM,
                                               BugReporter &BR) const {

  AnalysisDeclContext *ADC = AM.getAnalysisDeclContext(D);

  auto MatchExprPtr = allOf(
      hasSourceExpression(hasTypePointingTo(cxxRecordDecl().bind(BaseNode))),
      hasTypePointingTo(cxxRecordDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
                            .bind(DerivedNode)));
  auto MatchExprPtrObjC = allOf(
      hasSourceExpression(ignoringImpCasts(hasType(objcObjectPointerType(
          pointee(hasDeclaration(objcInterfaceDecl().bind(BaseNode))))))),
      ignoringImpCasts(hasType(objcObjectPointerType(pointee(hasDeclaration(
          objcInterfaceDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
              .bind(DerivedNode)))))));
  auto MatchExprRef =
      allOf(hasSourceExpression(hasType(cxxRecordDecl().bind(BaseNode))),
            hasType(cxxRecordDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
                        .bind(DerivedNode)));
  auto MatchExprRefTypeDef =
      allOf(hasSourceExpression(hasType(hasUnqualifiedDesugaredType(recordType(
                 hasDeclaration(decl(cxxRecordDecl().bind(BaseNode))))))),
            hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                decl(cxxRecordDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
                         .bind(DerivedNode)))))));

  auto CastC = cStyleCastExpr(anyOf(MatchExprPtr, MatchExprRef,
                                    MatchExprRefTypeDef, MatchExprPtrObjC))
                   .bind(WarnRecordDecl);
  auto CastStatic =
      cxxStaticCastExpr(anyOf(MatchExprPtr, MatchExprRef, MatchExprRefTypeDef,
                              MatchExprPtrObjC))
          .bind(WarnRecordDecl);
  auto CastReinterpret =
      cxxReinterpretCastExpr(anyOf(MatchExprPtr, MatchExprRef,
                                   MatchExprRefTypeDef, MatchExprPtrObjC))
          .bind(WarnRecordDecl);
  auto CastDynamic =
      cxxDynamicCastExpr(anyOf(MatchExprPtr, MatchExprRef, MatchExprRefTypeDef,
                               MatchExprPtrObjC))
          .bind(WarnRecordDecl);

  auto Cast = stmt(anyOf(CastC, CastStatic, CastReinterpret, CastDynamic));

  auto Matches =
      match(stmt(forEachDescendant(Cast)), *D->getBody(), AM.getASTContext());
  for (BoundNodes Match : Matches)
    emitDiagnostics(Match, BR, ADC, this);
}

void ento::registerMemoryUnsafeCastChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<MemoryUnsafeCastChecker>();
}

bool ento::shouldRegisterMemoryUnsafeCastChecker(const CheckerManager &mgr) {
  return true;
}
