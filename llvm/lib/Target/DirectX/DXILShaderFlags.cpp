//===- DXILShaderFlags.cpp - DXIL Shader Flags helper objects -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects and APIs for working with DXIL
///       Shader Flags.
///
//===----------------------------------------------------------------------===//

#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dxil;

static void updateFunctionFlags(ComputedShaderFlags &CSF,
                                const Instruction &I) {
  if (!CSF.Doubles)
    CSF.Doubles = I.getType()->isDoubleTy();

  if (!CSF.Doubles) {
    for (Value *Op : I.operands())
      CSF.Doubles |= Op->getType()->isDoubleTy();
  }
  if (CSF.Doubles) {
    switch (I.getOpcode()) {
    case Instruction::FDiv:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      // TODO: To be set if I is a call to DXIL intrinsic DXIL::Opcode::Fma
      // https://github.com/llvm/llvm-project/issues/114554
      CSF.DX11_1_DoubleExtensions = true;
      break;
    }
  }
}

void ModuleShaderFlags::initialize(const Module &M) {
  SmallVector<const Function *> WorkList;
  // Collect shader flags for each of the functions
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    if (!F.user_empty()) {
      WorkList.push_back(&F);
    }
    ComputedShaderFlags CSF;
    for (const auto &BB : F)
      for (const auto &I : BB)
        updateFunctionFlags(CSF, I);
    // Insert shader flag mask for function F
    FunctionFlags.push_back({&F, CSF});
    // Update combined shader flags mask
    CombinedSFMask.merge(CSF);
  }
  llvm::sort(FunctionFlags);
  // Propagate shader flag mask of functions to their callers.
  while (!WorkList.empty()) {
    const Function *Func = WorkList.pop_back_val();
    if (!Func->user_empty()) {
      ComputedShaderFlags FuncSF = getFunctionFlags(Func);
      // Update mask of callers with that of Func
      for (const auto User : Func->users()) {
        if (const CallInst *CI = dyn_cast<CallInst>(User)) {
          const Function *Caller = CI->getParent()->getParent();
          if (mergeFunctionShaderFlags(Caller, FuncSF))
            WorkList.push_back(Caller);
        }
      }
    }
  }
}

void ComputedShaderFlags::print(raw_ostream &OS) const {
  uint64_t FlagVal = (uint64_t) * this;
  OS << formatv("; Shader Flags Value: {0:x8}\n;\n", FlagVal);
  if (FlagVal == 0)
    return;
  OS << "; Note: shader requires additional functionality:\n";
#define SHADER_FEATURE_FLAG(FeatureBit, DxilModuleNum, FlagName, Str)          \
  if (FlagName)                                                                \
    (OS << ";").indent(7) << Str << "\n";
#include "llvm/BinaryFormat/DXContainerConstants.def"
  OS << "; Note: extra DXIL module flags:\n";
#define DXIL_MODULE_FLAG(DxilModuleBit, FlagName, Str)                         \
  if (FlagName)                                                                \
    (OS << ";").indent(7) << Str << "\n";
#include "llvm/BinaryFormat/DXContainerConstants.def"
  OS << ";\n";
}

auto ModuleShaderFlags::getFunctionShaderFlagInfo(const Function *Func) const {
  const auto Iter = llvm::lower_bound(
      FunctionFlags, Func,
      [](const std::pair<const Function *, ComputedShaderFlags> FSM,
         const Function *FindFunc) { return (FSM.first < FindFunc); });
  assert((Iter != FunctionFlags.end() && Iter->first == Func) &&
         "No Shader Flags Mask exists for function");
  return Iter;
}

/// Merge mask NewSF to that of Func, if different.
/// Return true if mask of Func is changed, else false.
bool ModuleShaderFlags::mergeFunctionShaderFlags(
    const Function *Func, const ComputedShaderFlags NewSF) {
  const auto FuncSFInfo = getFunctionShaderFlagInfo(Func);
  if ((FuncSFInfo->second & NewSF) != NewSF) {
    const_cast<ComputedShaderFlags &>(FuncSFInfo->second).merge(NewSF);
    return true;
  }
  return false;
}
/// Return the shader flags mask of the specified function Func.
const ComputedShaderFlags &
ModuleShaderFlags::getFunctionFlags(const Function *Func) const {
  return getFunctionShaderFlagInfo(Func)->second;
}

//===----------------------------------------------------------------------===//
// ShaderFlagsAnalysis and ShaderFlagsAnalysisPrinterPass

// Provide an explicit template instantiation for the static ID.
AnalysisKey ShaderFlagsAnalysis::Key;

ModuleShaderFlags ShaderFlagsAnalysis::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  ModuleShaderFlags MSFI;
  MSFI.initialize(M);
  return MSFI;
}

PreservedAnalyses ShaderFlagsAnalysisPrinter::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  const ModuleShaderFlags &FlagsInfo = AM.getResult<ShaderFlagsAnalysis>(M);
  // Print description of combined shader flags for all module functions
  OS << "; Combined Shader Flags for Module\n";
  FlagsInfo.getCombinedFlags().print(OS);
  // Print shader flags mask for each of the module functions
  OS << "; Shader Flags for Module Functions\n";
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    auto SFMask = FlagsInfo.getFunctionFlags(&F);
    OS << formatv("; Function {0} : {1:x8}\n;\n", F.getName(),
                  (uint64_t)(SFMask));
  }

  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// ShaderFlagsAnalysis and ShaderFlagsAnalysisPrinterPass

bool ShaderFlagsAnalysisWrapper::runOnModule(Module &M) {
  MSFI.initialize(M);
  return false;
}

char ShaderFlagsAnalysisWrapper::ID = 0;

INITIALIZE_PASS(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                "DXIL Shader Flag Analysis", true, true)
