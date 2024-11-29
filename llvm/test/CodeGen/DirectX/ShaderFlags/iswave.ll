; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Shader Flags Value: 0x00080000
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Wave level operations
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: {{^;$}}

; TODO: is there a way that we can split the input file into many?
; as the shader flag will be true if any are set

define i32 @wrla(i32 %expr, i32 %idx) {
  %ret = call i32 @llvm.dx.wave.readlane.i32(i32 %expr, i32 %idx)
  ret i32 %ret
}

define i1 @waat(i1 %x) {
  %ret = call i1 @llvm.dx.wave.any(i1 %x)
  ret i1 %ret
}

define i32 @wabc(i1 %x) {
  %ret = call i32 @llvm.dx.wave.active.countbits(i1 %x)
  ret i32 %ret
}

define i1 @wifl() {
  %ret = call i1 @llvm.dx.wave.is.first.lane()
  ret i1 %ret
}

define i32 @wgli() {
  %ret = call i32 @llvm.dx.wave.getlaneindex()
  ret i32 %ret
}
