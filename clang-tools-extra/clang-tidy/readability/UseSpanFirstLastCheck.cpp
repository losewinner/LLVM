//===--- UseSpanFirstLastCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseSpanFirstLastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
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
  unsigned NumArgs = Call->getNumArgs();
  if (NumArgs == 0 || NumArgs > 2)
    return;

  const Expr *Offset = Call->getArg(0);
  const Expr *Count = NumArgs > 1 ? Call->getArg(1) : nullptr;
  auto &Context = *Result.Context;

  class SubspanVisitor : public RecursiveASTVisitor<SubspanVisitor> {
  public:
    SubspanVisitor(const ASTContext &Context) : Context(Context) {}

    TraversalKind getTraversalKind() const {
      return TK_IgnoreUnlessSpelledInSource;
    }

    bool VisitIntegerLiteral(IntegerLiteral *IL) {
      if (IL->getValue() == 0)
        IsZeroOffset = true;
      return true;
    }

    bool VisitBinaryOperator(BinaryOperator *BO) {
      if (BO->getOpcode() == BO_Sub) {
        if (const auto *SizeCall = dyn_cast<CXXMemberCallExpr>(BO->getLHS())) {
          if (SizeCall->getMethodDecl()->getName() == "size") {
            IsSizeMinusN = true;
            SizeMinusArg = BO->getRHS();
          }
        }
      }
      return true;
    }

    bool IsZeroOffset = false;
    bool IsSizeMinusN = false;
    const Expr *SizeMinusArg = nullptr;

  private:
    const ASTContext &Context;
  };

  SubspanVisitor Visitor(Context);
  Visitor.TraverseStmt(const_cast<Expr *>(Offset->IgnoreImpCasts()));

  // Build replacement text
  std::string Replacement;
  if (Visitor.IsZeroOffset && Count) {
    // subspan(0, count) -> first(count)
    const StringRef CountStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Count->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    const auto *Base =
        cast<CXXMemberCallExpr>(Call)->getImplicitObjectArgument();
    const StringRef BaseStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Base->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    Replacement = BaseStr.str() + ".first(" + CountStr.str() + ")";
  } else if (Visitor.IsSizeMinusN && Visitor.SizeMinusArg) {
    // subspan(size() - n) -> last(n)
    const StringRef ArgStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Visitor.SizeMinusArg->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    const auto *Base =
        cast<CXXMemberCallExpr>(Call)->getImplicitObjectArgument();
    const StringRef BaseStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Base->getSourceRange()),
        Context.getSourceManager(), Context.getLangOpts());
    Replacement = BaseStr.str() + ".last(" + ArgStr.str() + ")";
  }

  if (!Replacement.empty()) {
    if (Visitor.IsZeroOffset && Count) {
      diag(Call->getBeginLoc(), "prefer 'span::first()' over 'subspan()'")
          << FixItHint::CreateReplacement(Call->getSourceRange(), Replacement);
    } else {
      diag(Call->getBeginLoc(), "prefer 'span::last()' over 'subspan()'")
          << FixItHint::CreateReplacement(Call->getSourceRange(), Replacement);
    }
  }
}

} // namespace clang::tidy::readability
