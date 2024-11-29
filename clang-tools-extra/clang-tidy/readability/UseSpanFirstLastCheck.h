//===--- UseSpanFirstLastCheck.h - clang-tidy-------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USESPANFIRSTLASTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USESPANFIRSTLASTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Converts std::span::subspan() calls to the more modern first()/last()
/// methods where applicable.
///
/// For example:
/// \code
///   std::span<int> s = ...;
///   auto sub = s.subspan(0, n);    // ->  auto sub = s.first(n);
///   auto sub2 = s.subspan(n);      // ->  auto sub2 = s.last(s.size() - n);
/// \endcode
class UseSpanFirstLastCheck : public ClangTidyCheck {
public:
  UseSpanFirstLastCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void handleSubspanCall(const ast_matchers::MatchFinder::MatchResult &Result,
                         const CXXMemberCallExpr *Call);
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USESPANFIRSTLASTCHECK_H