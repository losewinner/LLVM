//===--- BuiltinsSPIRV.td - SPIRV Builtin function database ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def HLSLDistance : LangBuiltin<"HLSL_LANG"> {
  let Spellings = ["__builtin_hlsl_distance"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}

def HLSLLength : LangBuiltin<"HLSL_LANG"> {
  let Spellings = ["__builtin_hlsl_length"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void(...)";
}
