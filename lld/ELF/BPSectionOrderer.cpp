//===- BPSectionOrderer.cpp--------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPSectionOrderer.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/SectionOrderer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/BalancedPartitioning.h"
#include "llvm/Support/TimeProfiler.h"

#include "SymbolTable.h"
#include "Symbols.h"

using namespace llvm;
using namespace lld::elf;

llvm::DenseMap<const lld::elf::InputSectionBase *, int>
lld::elf::runBalancedPartitioning(Ctx &ctx, llvm::StringRef profilePath,
                                  bool forFunctionCompression,
                                  bool forDataCompression,
                                  bool compressionSortStartupFunctions,
                                  bool verbose) {
  size_t highestAvailablePriority = std::numeric_limits<int>::max();
  SmallVector<lld::BPSectionBase *> sections;

  for (Symbol *sym : ctx.symtab->getSymbols()) {
    if (auto *d = dyn_cast<Defined>(sym)) {
      if (auto *sec = dyn_cast_or_null<InputSectionBase>(d->section)) {
        sections.push_back(new ELFSection(sec, new ELFSymbol(sym)));
      }
    }
  }

  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *sym : file->getLocalSymbols()) {
      if (auto *d = dyn_cast<Defined>(sym)) {
        if (auto *sec = dyn_cast_or_null<InputSectionBase>(d->section)) {
          sections.push_back(new ELFSection(sec, new ELFSymbol(sym)));
        }
      }
    }

  auto reorderedSections =
      lld::SectionOrderer::reorderSectionsByBalancedPartitioning(
          highestAvailablePriority, profilePath, forFunctionCompression,
          forDataCompression, compressionSortStartupFunctions, verbose,
          sections);

  DenseMap<const InputSectionBase *, int> result;
  for (const auto &[BPSectionBase, priority] : reorderedSections) {
    if (const ELFSection *elfSection = dyn_cast<ELFSection>(BPSectionBase)) {
      result[elfSection->getSymbol()->getInputSection()] =
          static_cast<int>(priority);
      delete const_cast<ELFSection *>(elfSection)->getSymbol();
      delete const_cast<ELFSection *>(elfSection);
    }
  }
  return result;
}
