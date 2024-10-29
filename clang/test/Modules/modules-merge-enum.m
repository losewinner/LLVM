// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang -fmodules -fmodules-cache-path=%t/modcache -fsyntax-only %t/source.m -Xclang -ast-dump-all
// TODO: Expect some AST shape here.


//--- shared.h
// This header is shared between two modules, but it's not a module itself.
// The enums defined here are parsed in both modules, and merged while building ModB.
typedef enum MyEnum1 { MyVal_A } MyEnum1;
enum MyEnum2 { MyVal_B };
typedef enum { MyVal_C } MyEnum3;

// In this case, no merging happens on the EnumDecl in Objective-C, and ASTWriter writes both EnumConstantDecls when building ModB.
enum { MyVal_D };

//--- module.modulemap
module ModA {
  module ModAFile1 {
    header "ModAFile1.h"
    export *
  }
  module ModAFile2 {
    header "ModAFile2.h"
    export *
  }
}
// The goal of writing ModB is to test that ASTWriter can handle the merged AST nodes correctly.
// For example, a stale declaration in IdResolver can cause an assertion failure while writing the identifier table.
module ModB {
  header "ModBFile.h"
  export *
}

//--- ModAFile1.h
#include "shared.h"

//--- ModAFile2.h
// Including this file, triggers loading of the module ModA with nodes coming ModAFile1.h being hidden.

//--- ModBFile.h
// ModBFile depends on ModAFile2.h only.
#include "ModAFile2.h"
// Including shared.h here causes Sema to merge the AST nodes from shared.h with the hidden ones from ModA.
#include "shared.h"


//--- source.m
#include "ModBFile.h"

int main() { return MyVal_A + MyVal_B + MyVal_C + MyVal_D; }
