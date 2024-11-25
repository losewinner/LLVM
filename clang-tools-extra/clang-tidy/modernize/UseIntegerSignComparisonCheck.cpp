//===--- UseIntegerSignComparisonCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseIntegerSignComparisonCheck.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang::tidy::modernize {
static BindableMatcher<clang::Stmt>
intCastExpression(bool IsSigned, const std::string &CastBindName) {
  auto IntTypeExpr = expr(hasType(hasCanonicalType(qualType(
      isInteger(), IsSigned ? isSignedInteger() : unless(isSignedInteger())))));

  const auto ImplicitCastExpr =
      implicitCastExpr(hasSourceExpression(IntTypeExpr)).bind(CastBindName);

  const auto CStyleCastExpr = cStyleCastExpr(has(ImplicitCastExpr));
  const auto StaticCastExpr = cxxStaticCastExpr(has(ImplicitCastExpr));
  const auto FunctionalCastExpr = cxxFunctionalCastExpr(has(ImplicitCastExpr));

  return expr(anyOf(ImplicitCastExpr, CStyleCastExpr, StaticCastExpr,
                    FunctionalCastExpr));
}

static StringRef parseOpCode(BinaryOperator::Opcode Code) {
  switch (Code) {
  case BO_LT:
    return "cmp_less";
  case BO_GT:
    return "cmp_greater";
  case BO_LE:
    return "cmp_less_equal";
  case BO_GE:
    return "cmp_greater_equal";
  case BO_EQ:
    return "cmp_equal";
  case BO_NE:
    return "cmp_not_equal";
  default:
    return "";
  }
}

UseIntegerSignComparisonCheck::UseIntegerSignComparisonCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void UseIntegerSignComparisonCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

void UseIntegerSignComparisonCheck::registerMatchers(MatchFinder *Finder) {
  const auto SignedIntCastExpr = intCastExpression(true, "sIntCastExpression");
  const auto UnSignedIntCastExpr =
      intCastExpression(false, "uIntCastExpression");

  // Flag all operators "==", "<=", ">=", "<", ">", "!="
  // that are used between signed/unsigned
  const auto CompareOperator =
      binaryOperator(hasAnyOperatorName("==", "<=", ">=", "<", ">", "!="),
                     hasOperands(SignedIntCastExpr, UnSignedIntCastExpr),
                     unless(isInTemplateInstantiation()))
          .bind("intComparison");

  Finder->addMatcher(CompareOperator, this);
}

void UseIntegerSignComparisonCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseIntegerSignComparisonCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *SignedCastExpression =
      Result.Nodes.getNodeAs<ImplicitCastExpr>("sIntCastExpression");
  const auto *UnSignedCastExpression =
      Result.Nodes.getNodeAs<ImplicitCastExpr>("uIntCastExpression");
  assert(SignedCastExpression);
  assert(UnSignedCastExpression);

  // Ignore the match if we know that the signed int value is not negative.
  Expr::EvalResult EVResult;
  if (!SignedCastExpression->isValueDependent() &&
      SignedCastExpression->getSubExpr()->EvaluateAsInt(EVResult,
                                                        *Result.Context)) {
    const llvm::APSInt SValue = EVResult.Val.getInt();
    if (SValue.isNonNegative())
      return;
  }

  const auto *BinaryOp =
      Result.Nodes.getNodeAs<BinaryOperator>("intComparison");
  if (BinaryOp == nullptr)
    return;

  const BinaryOperator::Opcode OpCode = BinaryOp->getOpcode();

  const Expr *LHS = BinaryOp->getLHS()->IgnoreImpCasts();
  const Expr *RHS = BinaryOp->getRHS()->IgnoreImpCasts();
  if (LHS == nullptr || RHS == nullptr)
    return;
  const Expr *SubExprLHS = nullptr;
  const Expr *SubExprRHS = nullptr;
  SourceRange R1 = SourceRange(LHS->getBeginLoc());
  SourceRange R2 = SourceRange(BinaryOp->getOperatorLoc());
  SourceRange R3 = SourceRange(Lexer::getLocForEndOfToken(
      RHS->getEndLoc(), 0, *Result.SourceManager, getLangOpts()));
  if (const auto *LHSCast = llvm::dyn_cast<ExplicitCastExpr>(LHS)) {
    SubExprLHS = LHSCast->getSubExpr();
    R1 = SourceRange(LHS->getBeginLoc(),
                     SubExprLHS->getBeginLoc().getLocWithOffset(-1));
    R2.setBegin(Lexer::getLocForEndOfToken(
        SubExprLHS->getEndLoc(), 0, *Result.SourceManager, getLangOpts()));
  }
  if (const auto *RHSCast = llvm::dyn_cast<ExplicitCastExpr>(RHS)) {
    SubExprRHS = RHSCast->getSubExpr();
    R2.setEnd(SubExprRHS->getBeginLoc().getLocWithOffset(-1));
  }
  DiagnosticBuilder Diag =
      diag(BinaryOp->getBeginLoc(),
           "comparison between 'signed' and 'unsigned' integers");
  if (!getLangOpts().CPlusPlus20)
    return;
  const std::string CmpNamespace = ("std::" + parseOpCode(OpCode)).str();
  const std::string CmpHeader = "<utility>";
  // Prefer modernize-use-integer-sign-comparison when C++20 is available!
  Diag << FixItHint::CreateReplacement(
      CharSourceRange(R1, SubExprLHS != nullptr),
      llvm::Twine(CmpNamespace + "(").str());
  Diag << FixItHint::CreateReplacement(R2, ",");
  Diag << FixItHint::CreateReplacement(CharSourceRange::getCharRange(R3), ")");

  // If there is no include for cmp_{*} functions, we'll add it.
  Diag << IncludeInserter.createIncludeInsertion(
      Result.SourceManager->getFileID(BinaryOp->getBeginLoc()), CmpHeader);
}

} // namespace clang::tidy::modernize
