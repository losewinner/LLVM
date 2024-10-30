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
static BindableMatcher<clang::Stmt> intCastExpression(bool IsSigned,
                                                      const std::string &CastBindName) {
  auto IntTypeExpr = IsSigned
                         ? expr(hasType(hasCanonicalType(qualType(isInteger(), isSignedInteger()))))
                         : expr(hasType(hasCanonicalType(qualType(isInteger(), unless(isSignedInteger())))));

  const auto ImplicitCastExpr =
      implicitCastExpr(hasSourceExpression(IntTypeExpr)).bind(CastBindName);

  const auto CStyleCastExpr = cStyleCastExpr(has(ImplicitCastExpr));
  const auto StaticCastExpr = cxxStaticCastExpr(has(ImplicitCastExpr));
  const auto FunctionalCastExpr = cxxFunctionalCastExpr(has(ImplicitCastExpr));

  return expr(anyOf(ImplicitCastExpr, CStyleCastExpr,
                    StaticCastExpr, FunctionalCastExpr));
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
                      areDiagsSelfContained()),
      IsQtApplication(Options.get("IsQtApplication", false)) {}

void UseIntegerSignComparisonCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  Options.store(Opts, "IsQtApplication", IsQtApplication);
}

void UseIntegerSignComparisonCheck::registerMatchers(MatchFinder *Finder) {
  const auto SignedIntCastExpr = intCastExpression(true, "sIntCastExpression");
  const auto UnSignedIntCastExpr =
      intCastExpression(false, "uIntCastExpression");

  // Flag all operators "==", "<=", ">=", "<", ">", "!="
  // that are used between signed/unsigned
  const auto CompareOperator =
      binaryOperator(hasAnyOperatorName("==", "<=", ">=", "<", ">", "!="),
                     hasOperands(SignedIntCastExpr,
                                 UnSignedIntCastExpr),
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
  const Expr *LHS = BinaryOp->getLHS()->IgnoreParenImpCasts();
  const Expr *RHS = BinaryOp->getRHS()->IgnoreParenImpCasts();
  if (LHS == nullptr || RHS == nullptr)
    return;

  const Expr *ExplicitLHS = nullptr;
  const Expr *ExplicitRHS = nullptr;
  StringRef ExplicitLhsString, ExplicitLhsRhsString;

  DiagnosticBuilder Diag = diag(BinaryOp->getBeginLoc(), "comparison between 'signed' and 'unsigned' integers");
  if (BinaryOp->getLHS() == SignedCastExpression)
  {
    ExplicitLHS = SignedCastExpression->isPartOfExplicitCast() ? SignedCastExpression : nullptr;
    ExplicitRHS = UnSignedCastExpression->isPartOfExplicitCast() ? UnSignedCastExpression : nullptr;
  } else {
    ExplicitRHS = SignedCastExpression->isPartOfExplicitCast() ? SignedCastExpression : nullptr;
    ExplicitLHS = UnSignedCastExpression->isPartOfExplicitCast() ? UnSignedCastExpression : nullptr;
  }

  if (!(getLangOpts().CPlusPlus17 && IsQtApplication) &&
      !getLangOpts().CPlusPlus20)
    return;

  std::string CmpNamespace;
  std::string CmpHeader;
  if (getLangOpts().CPlusPlus20) {
    CmpNamespace = std::string("std::") + std::string(parseOpCode(OpCode));
    CmpHeader = std::string("<utility>");
  } else if (getLangOpts().CPlusPlus17 && IsQtApplication) {
    CmpNamespace = std::string("q20::") + std::string(parseOpCode(OpCode));
    CmpHeader = std::string("<QtCore/q20utility.h>");
  }

  // Use qt-use-integer-sign-comparison when C++17 is available and only for Qt
  // apps. Prefer modernize-use-integer-sign-comparison when C++20 is available!
  if (ExplicitLHS) {
    ExplicitLhsString = Lexer::getSourceText(CharSourceRange::getTokenRange(ExplicitLHS->IgnoreCasts()->getSourceRange()), *Result.SourceManager, getLangOpts());
    Diag << FixItHint::CreateReplacement(LHS->getSourceRange(), llvm::Twine(CmpNamespace + "(" + ExplicitLhsString).str());
  } else {
    Diag << FixItHint::CreateInsertion(LHS->getBeginLoc(),
                                       llvm::Twine(CmpNamespace + "(").str());
  }
  Diag << FixItHint::CreateReplacement(BinaryOp->getOperatorLoc(), ", ");
  if (ExplicitRHS) {
    ExplicitLhsRhsString = Lexer::getSourceText(CharSourceRange::getTokenRange(ExplicitRHS->IgnoreCasts()->getSourceRange()), *Result.SourceManager, getLangOpts());
    Diag << FixItHint::CreateReplacement(RHS->getSourceRange(), llvm::Twine(ExplicitLhsRhsString + ")").str());
  } else {
    Diag << FixItHint::CreateInsertion(
        Lexer::getLocForEndOfToken(
            Result.SourceManager->getSpellingLoc(RHS->getEndLoc()), 0,
            *Result.SourceManager, Result.Context->getLangOpts()),
        ")");
  }

  // If there is no include for cmp_{*} functions, we'll add it.
  Diag << IncludeInserter.createIncludeInsertion(
      Result.SourceManager->getFileID(BinaryOp->getBeginLoc()),
      CmpHeader);
}

} // namespace clang::tidy::modernize
