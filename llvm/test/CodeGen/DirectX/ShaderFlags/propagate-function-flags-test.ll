; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000044
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Double-precision floating point
; CHECK-NEXT: ;       Double-precision extensions for 11.1
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

; CHECK: ; Function call_n6 : 0x00000044
define double @call_n6(i32 noundef %0) local_unnamed_addr #0 {
  %2 = sitofp i32 %0 to double
  ret double %2
}
; CHECK: ; Function call_n4 : 0x00000044
define double @call_n4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @call_n6(i32 noundef %0)
  ret double %2
}

; CHECK: ; Function call_n7 : 0x00000044
define double @call_n7(i32 noundef %0) local_unnamed_addr #0 {
  %2 = uitofp i32 %0 to double
  ret double %2
}

; CHECK: ; Function call_n5 : 0x00000044
define double @call_n5(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @call_n7(i32 noundef %0)
  ret double %2
}

; CHECK: ; Function call_n2 : 0x00000044
define double @call_n2(i64 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i64 %0, 6
  br i1 %2, label %3, label %7

3:                                                ; preds = %1
  %4 = add nuw nsw i64 %0, 1
  %5 = uitofp i64 %4 to double
  %6 = tail call double @call_n1(double noundef %5)
  br label %10

7:                                                ; preds = %1
  %8 = trunc i64 %0 to i32
  %9 = tail call double @call_n4(i32 noundef %8)
  br label %10

10:                                               ; preds = %7, %3
  %11 = phi double [ %6, %3 ], [ %9, %7 ]
  ret double %11
}

; CHECK: ; Function call_n1 : 0x00000044
define double @call_n1(double noundef %0) local_unnamed_addr #0 {
  %2 = fcmp ugt double %0, 5.000000e+00
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = fptoui double %0 to i64
  %5 = tail call double @call_n2(i64 noundef %4)
  br label %9

6:                                                ; preds = %1
  %7 = fptoui double %0 to i32
  %8 = tail call double @call_n5(i32 noundef %7)
  br label %9

9:                                                ; preds = %6, %3
  %10 = phi double [ %5, %3 ], [ %8, %6 ]
  ret double %10
}

; CHECK: ; Function call_n3 : 0x00000044
define double @call_n3(double noundef %0) local_unnamed_addr #0 {
  %2 = fdiv double %0, 3.000000e+00
  ret double %2
}

; CHECK: ; Function main : 0x00000044
define i32 @main() local_unnamed_addr #0 {
  %1 = tail call double @call_n1(double noundef 1.000000e+00)
  %2 = tail call double @call_n3(double noundef %1)
  ret i32 0
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
