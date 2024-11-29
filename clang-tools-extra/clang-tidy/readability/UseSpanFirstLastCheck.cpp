//===--- UseSpanFirstLastCheck.cpp - clang-tidy-----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseSpanFirstLastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void UseSpanFirstLastCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus20)
    return;

  // Match span::subspan calls
  const auto HasSpanType =
      hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
          classTemplateSpecializationDecl(hasName("::std::span"))))));

  Finder->addMatcher(cxxMemberCallExpr(callee(memberExpr(hasDeclaration(
                                           cxxMethodDecl(hasName("subspan"))))),
                                       on(expr(HasSpanType)))
                         .bind("subspan"),
                     this);
}

void UseSpanFirstLastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("subspan");
  if (!Call)
    return;

  handleSubspanCall(Result, Call);
}

void UseSpanFirstLastCheck::handleSubspanCall(
    const MatchFinder::MatchResult &Result, const CXXMemberCallExpr *Call) {
  unsigned NumArgs = Call->getNumArgs();
  if (NumArgs == 0 || NumArgs > 2)
    return;

  const Expr *Offset = Call->getArg(0);
  const Expr *Count = NumArgs > 1 ? Call->getArg(1) : nullptr;
  auto &Context = *Result.Context;

  // Check if this is subspan(0, n) -> first(n)
  bool IsZeroOffset = false;
  const Expr *OffsetE = Offset->IgnoreImpCasts();
  if (const auto *IL = dyn_cast<IntegerLiteral>(OffsetE)) {
    IsZeroOffset = IL->getValue() == 0;
  }

  // Check if this is subspan(size() - n) -> last(n)
  bool IsSizeMinusN = false;
  const Expr *SizeMinusArg = nullptr;
  if (const auto *BO = dyn_cast<BinaryOperator>(OffsetE)) {
    if (BO->getOpcode() == BO_Sub) {
      if (const auto *SizeCall = dyn_cast<CXXMemberCallExpr>(BO->getLHS())) {
        if (SizeCall->getMethodDecl()->getName() == "size") {
          IsSizeMinusN = true;
          SizeMinusArg = BO->getRHS();
        }
      }
    }
  }

  // Build replacement text
  std::string Replacement;
  if (IsZeroOffset && Count) {
    // subspan(0, count) -> first(count)
    auto CountStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Count->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    const auto *Base =
        cast<CXXMemberCallExpr>(Call)->getImplicitObjectArgument();
    auto BaseStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Base->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    Replacement = BaseStr.str() + ".first(" + CountStr.str() + ")";
  } else if (IsSizeMinusN && SizeMinusArg) {
    // subspan(size() - n) -> last(n)
    auto ArgStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(SizeMinusArg->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    const auto *Base =
        cast<CXXMemberCallExpr>(Call)->getImplicitObjectArgument();
    auto BaseStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Base->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    Replacement = BaseStr.str() + ".last(" + ArgStr.str() + ")";
  }

  if (!Replacement.empty()) {
    if (IsZeroOffset && Count) {
      diag(Call->getBeginLoc(), "prefer span::first() over subspan()")
          << FixItHint::CreateReplacement(Call->getSourceRange(), Replacement);
    } else {
      diag(Call->getBeginLoc(), "prefer span::last() over subspan()")
          << FixItHint::CreateReplacement(Call->getSourceRange(), Replacement);
    }
  }
}
} // namespace clang::tidy::readability