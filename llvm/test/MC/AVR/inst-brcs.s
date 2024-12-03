; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brcs .+8
  brcs .+4
  brbs 0, .+8
  brbs 0, bar

bar:

; CHECK: brcs (.Ltmp0+8)+2  ; encoding: [0bAAAAA000,0b111100AA]
; CHECK-NEXT:               ;   fixup A - offset: 0, value: (.Ltmp0+8)+2, kind: fixup_7_pcrel
; CHECK: brcs (.Ltmp1+4)+2  ; encoding: [0bAAAAA000,0b111100AA]
; CHECK-NEXT:               ;   fixup A - offset: 0, value: (.Ltmp1+4)+2, kind: fixup_7_pcrel
; CHECK: brcs (.Ltmp2+8)+2  ; encoding: [0bAAAAA000,0b111100AA]
; CHECK-NEXT:               ;   fixup A - offset: 0, value: (.Ltmp2+8)+2, kind: fixup_7_pcrel
; CHECK: brcs bar           ; encoding: [0bAAAAA000,0b111100AA]
; CHECK-NEXT:               ;   fixup A - offset: 0, value: bar, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 00 f0 brlo .+0
; INST-NEXT: R_AVR_7_PCREL .text+0xa
; INST-NEXT: 00 f0 brlo .+0
; INST-NEXT: R_AVR_7_PCREL .text+0x8
; INST-NEXT: 00 f0 brlo .+0
; INST-NEXT: R_AVR_7_PCREL .text+0xe
; INST-NEXT: 00 f0 brlo .+0
; INST-NEXT: R_AVR_7_PCREL .text+0x8
