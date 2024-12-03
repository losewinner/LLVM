; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brcc .+66
  brcc .-22
  brbc 0, .+66
  brbc 0, bar

bar:

; CHECK: brcc (.Ltmp0+66)+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp0+66)+2, kind: fixup_7_pcrel
; CHECK: brcc (.Ltmp1-22)+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp1-22)+2, kind: fixup_7_pcrel
; CHECK: brcc (.Ltmp2+66)+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: (.Ltmp2+66)+2, kind: fixup_7_pcrel
; CHECK: brcc bar            ; encoding: [0bAAAAA000,0b111101AA]
; CHECK-NEXT:                ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 00 f4 brsh .+0
; INST-NEXT: R_AVR_7_PCREL .text+0x44
; INST-NEXT: 00 f4 brsh .+0
; INST-NEXT: R_AVR_7_PCREL .text-0x12
; INST-NEXT: 00 f4 brsh .+0
; INST-NEXT: R_AVR_7_PCREL .text+0x48
; INST-NEXT: 00 f4 brsh .+0
; INST-NEXT: R_AVR_7_PCREL .text+0x8
