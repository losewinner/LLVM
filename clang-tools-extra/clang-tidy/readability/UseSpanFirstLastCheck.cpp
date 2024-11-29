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
  // Get arguments
  unsigned NumArgs = Call->getNumArgs();
  if (NumArgs == 0 || NumArgs > 2)
    return;

  const Expr *Offset = Call->getArg(0);
  const Expr *Count = NumArgs > 1 ? Call->getArg(1) : nullptr;
  auto &Context = *Result.Context;
  bool IsZeroOffset = false;

  // Check if offset is zero through any implicit casts
  const Expr *OffsetE = Offset->IgnoreImpCasts();
  if (const auto *IL = dyn_cast<IntegerLiteral>(OffsetE)) {
    IsZeroOffset = IL->getValue() == 0;
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
  } else if (NumArgs == 1) {
    // subspan(n) -> last(size() - n)
    auto OffsetStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Offset->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());

    const auto *Base =
        cast<CXXMemberCallExpr>(Call)->getImplicitObjectArgument();
    auto BaseStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Base->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());

    Replacement = BaseStr.str() + ".last(" + BaseStr.str() + ".size() - " +
                  OffsetStr.str() + ")";
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