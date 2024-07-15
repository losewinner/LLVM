; RUN: opt -mattr=+sve2 -mtriple=aarch64 -passes='loop(loop-idiom-vectorize)' -S < %s | FileCheck -check-prefix=SVE2 %s
; RUN: opt -mattr=-sve2 -mtriple=aarch64 -passes='loop(loop-idiom-vectorize)' -S < %s | FileCheck -check-prefix=NOSVE2 %s

; Base case based on `libcxx/include/__algorithm/find_first_of.h':
;   char* find_first_of(char *first, char *last, char *s_first, char *s_last) {
;     for (; first != last; ++first)
;       for (char *it = s_first; it != s_last; ++it)
;         if (*first == *it)
;           return first;
;     return last;
;   }
define ptr @find_first_of_i8(ptr %0, ptr %1, ptr %2, ptr %3) #0 {
; SVE2-LABEL: define ptr @find_first_of_i8(
; SVE2:         {{%.*}} = call <vscale x 16 x i1> @llvm.experimental.vector.match.nxv16i8.v16i8(<vscale x 16 x i8> {{%.*}}, <16 x i8> {{%.*}}, <vscale x 16 x i1> {{%.*}})
;
; NOSVE2-LABEL: define ptr @find_first_of_i8(
; NOSVE2-NOT:     {{%.*}} = call <vscale x 16 x i1> @llvm.experimental.vector.match.nxv16i8.v16i8(<vscale x 16 x i8> {{%.*}}, <16 x i8> {{%.*}}, <vscale x 16 x i1> {{%.*}})
;
  %5 = icmp eq ptr %0, %1
  %6 = icmp eq ptr %2, %3
  %7 = or i1 %5, %6
  br i1 %7, label %21, label %8

8:
  %9 = phi ptr [ %19, %18 ], [ %0, %4 ]
  %10 = load i8, ptr %9, align 1
  br label %14

11:
  %12 = getelementptr inbounds i8, ptr %15, i64 1
  %13 = icmp eq ptr %12, %3
  br i1 %13, label %18, label %14

14:
  %15 = phi ptr [ %2, %8 ], [ %12, %11 ]
  %16 = load i8, ptr %15, align 1
  %17 = icmp eq i8 %10, %16
  br i1 %17, label %21, label %11

18:
  %19 = getelementptr inbounds i8, ptr %9, i64 1
  %20 = icmp eq ptr %19, %1
  br i1 %20, label %21, label %8

21:
  %22 = phi ptr [ %1, %4 ], [ %9, %14 ], [ %1, %18 ]
  ret ptr %22
}

; Same as @find_first_of_i8 but with i16.
define ptr @find_first_of_i16(ptr %0, ptr %1, ptr %2, ptr %3) #0 {
; SVE2-LABEL: define ptr @find_first_of_i16(
; SVE2:         {{%.*}} = call <vscale x 8 x i1> @llvm.experimental.vector.match.nxv8i16.v8i16(<vscale x 8 x i16> {{%.*}}, <8 x i16> {{%.*}}, <vscale x 8 x i1> {{%.*}})
;
  %5 = icmp eq ptr %0, %1
  %6 = icmp eq ptr %2, %3
  %7 = or i1 %5, %6
  br i1 %7, label %21, label %8

8:
  %9 = phi ptr [ %19, %18 ], [ %0, %4 ]
  %10 = load i16, ptr %9, align 1
  br label %14

11:
  %12 = getelementptr inbounds i16, ptr %15, i64 1
  %13 = icmp eq ptr %12, %3
  br i1 %13, label %18, label %14

14:
  %15 = phi ptr [ %2, %8 ], [ %12, %11 ]
  %16 = load i16, ptr %15, align 1
  %17 = icmp eq i16 %10, %16
  br i1 %17, label %21, label %11

18:
  %19 = getelementptr inbounds i16, ptr %9, i64 1
  %20 = icmp eq ptr %19, %1
  br i1 %20, label %21, label %8

21:
  %22 = phi ptr [ %1, %4 ], [ %9, %14 ], [ %1, %18 ]
  ret ptr %22
}

; Same as @find_first_of_i8 but with `ne' comparison.
; This is rejected for now, but should eventually be supported.
define ptr @find_first_not_of_i8(ptr %0, ptr %1, ptr %2, ptr %3) #0 {
; SVE2-LABEL: define ptr @find_first_not_of_i8(
; SVE2-NOT:     {{%.*}} = call <vscale x 16 x i1> @llvm.experimental.vector.match.nxv16i8.v16i8(<vscale x 16 x i8> {{%.*}}, <16 x i8> {{%.*}}, <vscale x 16 x i1> {{%.*}})
;
  %5 = icmp eq ptr %0, %1
  %6 = icmp eq ptr %2, %3
  %7 = or i1 %5, %6
  br i1 %7, label %21, label %8

8:
  %9 = phi ptr [ %19, %18 ], [ %0, %4 ]
  %10 = load i8, ptr %9, align 1
  br label %14

11:
  %12 = getelementptr inbounds i8, ptr %15, i64 1
  %13 = icmp eq ptr %12, %3
  br i1 %13, label %18, label %14

14:
  %15 = phi ptr [ %2, %8 ], [ %12, %11 ]
  %16 = load i8, ptr %15, align 1
  %17 = icmp ne i8 %10, %16
  br i1 %17, label %21, label %11

18:
  %19 = getelementptr inbounds i8, ptr %9, i64 1
  %20 = icmp eq ptr %19, %1
  br i1 %20, label %21, label %8

21:
  %22 = phi ptr [ %1, %4 ], [ %9, %14 ], [ %1, %18 ]
  ret ptr %22
}

attributes #0 = { "target-features"="+sve2" }
