# Check that using link_fdata tool in non-lbr mode allows using hardcoded
# addresses for basic block offsets.

# REQUIRES: system-linux

# RUN: %clang %cflags -o %t %s
# RUN: %clang %s %cflags -Wl,-q -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: cat %t.fdata | FileCheck %s

  .text
  .globl  foo
  .type foo, %function
foo:
# FDATA: 1 foo 0 10
    ret

# CHECK: no_lbr
# CHECK-NEXT: 1 foo 0 10
