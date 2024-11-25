//===- BPSectionOrderer.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file uses Balanced Partitioning to order sections to improve startup
/// time and compressed size.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_BPSECTION_ORDERER_H
#define LLD_ELF_BPSECTION_ORDERER_H

#include "InputFiles.h"
#include "InputSection.h"
#include "Relocations.h"
#include "Symbols.h"
#include "lld/Common/SectionOrderer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/xxhash.h"

namespace lld::elf {

class InputSection;

class ELFSymbol : public BPSymbol {
  const Symbol *sym;

public:
  explicit ELFSymbol(const Symbol *s) : sym(s) {}

  llvm::StringRef getName() const override { return sym->getName(); }

  BPSymbol *asDefinedSymbol() override {
    if (auto *d = llvm::dyn_cast<Defined>(sym))
      return this;
    return nullptr;
  }

  uint64_t getValue() const override {
    if (auto *d = llvm::dyn_cast<Defined>(sym))
      return d->value;
    return 0;
  }

  uint64_t getSize() const override {
    if (auto *d = llvm::dyn_cast<Defined>(sym))
      return d->size;
    return 0;
  }

  InputSectionBase *getInputSection() const {
    if (auto *d = llvm::dyn_cast<Defined>(sym))
      return llvm::dyn_cast_or_null<InputSectionBase>(d->section);
    return nullptr;
  }

  const Symbol *getSymbol() const { return sym; }
};

class ELFSection : public BPSectionBase {
  const InputSectionBase *isec;
  ELFSymbol *symbol;
  std::vector<BPSymbol *> symbols;

public:
  explicit ELFSection(const InputSectionBase *sec, ELFSymbol *sym)
      : isec(sec), symbol(sym), symbols({sym}) {}

  const InputSectionBase *getSection() const { return isec; }

  ELFSymbol *getSymbol() const { return symbol; }
  llvm::StringRef getName() const override { return isec->name; }

  uint64_t getSize() const override { return isec->getSize(); }

  bool isCodeSection() const override {
    return isec->flags & llvm::ELF::SHF_EXECINSTR;
  }

  bool hasValidData() const override {
    return isec && !isec->content().empty();
  }

  llvm::ArrayRef<uint8_t> getSectionData() const override {
    return isec->content();
  }

  llvm::ArrayRef<BPSymbol *> getSymbols() const override { return symbols; }

  void getSectionHash(llvm::SmallVectorImpl<uint64_t> &hashes,
                      const llvm::DenseMap<const BPSectionBase *, uint64_t>
                          &sectionToIdx) const override {
    constexpr unsigned windowSize = 4;

    // Convert BPSectionBase map to InputSection map
    llvm::DenseMap<const InputSectionBase *, uint64_t> elfSectionToIdx;
    for (const auto &[sec, idx] : sectionToIdx) {
      if (auto *elfSec = llvm::dyn_cast<ELFSection>(sec))
        elfSectionToIdx[elfSec->getSection()] = idx;
    }

    // Calculate content hashes
    for (size_t i = 0; i < isec->content().size(); i++) {
      auto window = isec->content().drop_front(i).take_front(windowSize);
      hashes.push_back(xxHash64(window));
    }

    // TODO: Calculate relocation hashes.
    // Since in ELF, relocations are complex, but the effect without them are
    // good enough, we just use 0 as their hash.

    llvm::sort(hashes);
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  }

  static bool classof(const BPSectionBase *s) { return true; }
};

/// Run Balanced Partitioning to find the optimal function and data order to
/// improve startup time and compressed size.
///
/// It is important that .subsections_via_symbols is used to ensure functions
/// and data are in their own sections and thus can be reordered.
llvm::DenseMap<const lld::elf::InputSectionBase *, int>
runBalancedPartitioning(Ctx &ctx, llvm::StringRef profilePath,
                        bool forFunctionCompression, bool forDataCompression,
                        bool compressionSortStartupFunctions, bool verbose);
} // namespace lld::elf

#endif
