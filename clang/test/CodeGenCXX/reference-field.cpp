// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s -O2 | FileCheck %s

// Make sure the call to b() is eliminated.
extern struct x {char& x,y;}y;
int b();      
int a() { if (!&y.x) b(); }

// CHECK-NOT: @_Z1bv
