# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: echo "A B 5" > %t.call_graph
# RUN: echo "B C 50" >> %t.call_graph
# RUN: echo "C D 40" >> %t.call_graph
# RUN: echo "D B 10" >> %t.call_graph
# RUN: not ld.lld %t --irpgo-profile-sort %s --call-graph-ordering-file=%t.call_graph 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# RUN: not ld.lld %t --irpgo-profile-sort=%s --call-graph-ordering-file=%t.call_graph 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# IRPGO-ERR: --irpgo-profile-sort is incompatible with --call-graph-ordering-file

# RUN: not ld.lld --compression-sort=function --call-graph-ordering-file %t.call_graph 2>&1 | FileCheck %s --check-prefix=COMPRESSION-ERR
# COMPRESSION-ERR: --compression-sort is incompatible with --call-graph-ordering-file

# RUN: not ld.lld --compression-sort=malformed 2>&1 | FileCheck %s --check-prefix=COMPRESSION-MALFORM
# COMPRESSION-MALFORM: unknown value 'malformed' for --compression-sort=

# RUN: not ld.lld --compression-sort-startup-functions 2>&1 | FileCheck %s --check-prefix=STARTUP
# STARTUP: --compression-sort-startup-functions must be used with --irpgo-profile-sort

# CHECK: B
# CHECK-NEXT: C
# CHECK-NEXT: D
# CHECK-NEXT: A
//--- a.s
.section    .text.A,"ax",@progbits
.globl  A
A:
 nop

.section    .text.B,"ax",@progbits
.globl  B
B:
 nop

.section    .text.C,"ax",@progbits
.globl  C
C:
 nop

.section    .text.D,"ax",@progbits
.globl  D
D:
 nop