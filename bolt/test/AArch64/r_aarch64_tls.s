## This test checks processing of tls relocations
##
## R_AARCH64_TLSLE_MOVW_TPREL_G0, TPREL(S + A)
## R_AARCH64_TLSLE_MOVW_TPREL_G0_NC, TPREL(S + A)

# REQUIRES: system-linux

# RUN: %clang %cflags -nostartfiles -nostdlib %s -o %t.exe -mlittle-endian \
# RUN:     -Wl,-q
# RUN: llvm-readelf -Wr %t.exe | FileCheck %s -check-prefix=CHECKTLS

# CHECKTLS:       R_AARCH64_TLSLE_MOVW_TPREL_G0      {{.*}} .tprel_tls_var + 0
# CHECKTLS-NEXT:  R_AARCH64_TLSLE_MOVW_TPREL_G0_NC   {{.*}} .tprel_tls_var + 4

# RUN: llvm-bolt %t.exe -o %t.bolt
# RUN: llvm-objdump -D %t.bolt | FileCheck %s --check-prefix=CHECKBOLT

# CHECKBOLT: Disassembly of section .tdata
# CHECKBOLT: [[#%x,DATATABLEADDR:]] <.tdata
# CHECKBOLT-NEXT: [[#DATATABLEADDR]]: 000000aa
# CHECKBOLT-NEXT: [[#DATATABLEADDR + 4]]: 000000bb

.section .tdata
.align 4
.tprel_tls_var:
.word 0xaa
.word 0xbb

.section .text
.align 4
.globl _start
.type _start, %function
_start:
  mrs x0, TPIDR_EL0
  movz x1, #:tprel_g0:.tprel_tls_var
  add x0, x0, x1
  ldr w2, [x0]
  cmp w2, 0xaa
  bne exit_failure

  mrs x0, TPIDR_EL0
  movk x1, #:tprel_g0_nc:.tprel_tls_var + 4
  add x0, x0, x1
  ldr w2, [x0]
  cmp w2, 0xbb
  bne exit_failure

exit_success:
  mov x0, #0
  b exit

exit_failure:
  mov x0, #1
  b exit

exit:
  mov x8, #93
  svc #0

.size _start, .-_start
