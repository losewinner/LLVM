# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: llvm-profdata merge %t/a.proftext -o %t/a.profdata

# The code below is the reduced version of the output
# from the following invocation and source:
#
# // test.cpp:
#     const char s1[] = "hello world";
#     const char s2[] = "i am a string"; 
#     const char *r1 = s1;
#     const char **r2 = &r1;
#
# void A()
# {
#     return;
# }
#
# int B(int a)
# {
#     A();
#     return a + 1;
# }
#
# int C(int a)
# {
#     A();
#     return a + 2;
# }
#
# int D(int a)
# {
#     return B(a + 2);
# }
#
# int E(int a)
# {
#     return C(a + 2);
# }
#
# int F(int a)
# {
#     return C(a + 3);
# }
#
# int main()
# {
#     return 0;
# }
#
# clang -target aarch64-linux-gnu -ffunction-sections -fdata-sections test.cpp -o test.s -S

# RUN: ld.lld -e main -o %t/a.out %t/a.o --irpgo-profile-sort=%t/a.profdata --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=STARTUP
# RUN: ld.lld -e main -o %t/a.out %t/a.o --irpgo-profile-sort=%t/a.profdata --verbose-bp-section-orderer --icf=all --compression-sort=none 2>&1 | FileCheck %s --check-prefix=STARTUP

# STARTUP: Ordered 3 sections using balanced partitioning

# RUN: ld.lld -e main -o - %t/a.o --irpgo-profile-sort=%t/a.profdata --symbol-ordering-file %t/a.orderfile | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=ORDERFILE

# ORDERFILE: _ZL2s1
# ORDERFILE: _ZL2s2
# ORDERFILE: _Z1Av
# ORDERFILE: _Z1Fi
# ORDERFILE: _Z1Ei
# ORDERFILE: _Z1Di
# ORDERFILE: _Z1Ci
# ORDERFILE: _Z1Bi
# ORDERFILE: main
# ORDERFILE: r1
# ORDERFILE: r2

# RUN: ld.lld -e main -o %t/a.out %t/a.o --verbose-bp-section-orderer --compression-sort=function 2>&1 | FileCheck %s --check-prefix=COMPRESSION-FUNC
# RUN: ld.lld -e main -o %t/a.out %t/a.o --verbose-bp-section-orderer --compression-sort=data 2>&1 | FileCheck %s --check-prefix=COMPRESSION-DATA
# RUN: ld.lld -e main -o %t/a.out %t/a.o --verbose-bp-section-orderer --compression-sort=both 2>&1 | FileCheck %s --check-prefix=COMPRESSION-BOTH
# RUN: ld.lld -e main -o %t/a.out %t/a.o --verbose-bp-section-orderer --compression-sort=both --irpgo-profile-sort=%t/a.profdata 2>&1 | FileCheck %s --check-prefix=COMPRESSION-BOTH

# COMPRESSION-FUNC: Ordered 7 sections using balanced partitioning
# COMPRESSION-DATA: Ordered 4 sections using balanced partitioning
# COMPRESSION-BOTH: Ordered 11 sections using balanced partitioning

#--- a.s
	.text
	.file	"test.cpp"
	.section	.text._Z1Av,"ax",@progbits
	.globl	_Z1Av                           // -- Begin function _Z1Av
	.p2align	2
	.type	_Z1Av,@function
_Z1Av:                                  // @_Z1Av
	.cfi_startproc
// %bb.0:
	ret
.Lfunc_end0:
	.size	_Z1Av, .Lfunc_end0-_Z1Av
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Bi,"ax",@progbits
	.globl	_Z1Bi                           // -- Begin function _Z1Bi
	.p2align	2
	.type	_Z1Bi,@function
_Z1Bi:                                  // @_Z1Bi
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
	bl	_Z1Av
	ldur	w8, [x29, #-4]
	add	w0, w8, #1
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end1:
	.size	_Z1Bi, .Lfunc_end1-_Z1Bi
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Ci,"ax",@progbits
	.globl	_Z1Ci                           // -- Begin function _Z1Ci
	.p2align	2
	.type	_Z1Ci,@function
_Z1Ci:                                  // @_Z1Ci
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
	bl	_Z1Av
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end2:
	.size	_Z1Ci, .Lfunc_end2-_Z1Ci
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Di,"ax",@progbits
	.globl	_Z1Di                           // -- Begin function _Z1Di
	.p2align	2
	.type	_Z1Di,@function
_Z1Di:                                  // @_Z1Di
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	bl	_Z1Bi
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end3:
	.size	_Z1Di, .Lfunc_end3-_Z1Di
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Ei,"ax",@progbits
	.globl	_Z1Ei                           // -- Begin function _Z1Ei
	.p2align	2
	.type	_Z1Ei,@function
_Z1Ei:                                  // @_Z1Ei
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	bl	_Z1Ci
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end4:
	.size	_Z1Ei, .Lfunc_end4-_Z1Ei
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Fi,"ax",@progbits
	.globl	_Z1Fi                           // -- Begin function _Z1Fi
	.p2align	2
	.type	_Z1Fi,@function
_Z1Fi:                                  // @_Z1Fi
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #3
	bl	_Z1Ci
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end5:
	.size	_Z1Fi, .Lfunc_end5-_Z1Fi
	.cfi_endproc
                                        // -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	mov	w0, wzr
	str	wzr, [sp, #12]
	add	sp, sp, #16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end6:
	.size	main, .Lfunc_end6-main
	.cfi_endproc
                                        // -- End function
	.type	_ZL2s1,@object                  // @_ZL2s1
	.section	.rodata._ZL2s1,"a",@progbits
_ZL2s1:
	.asciz	"hello world"
	.size	_ZL2s1, 12

	.type	r1,@object                      // @r1
	.section	.data.r1,"aw",@progbits
	.globl	r1
	.p2align	3, 0x0
r1:
	.xword	_ZL2s1
	.size	r1, 8

	.type	_ZL2s2,@object                  // @_ZL2s2
	.section	.rodata._ZL2s2,"a",@progbits
_ZL2s2:
	.asciz	"i am a string"
	.size	_ZL2s2, 14

	.type	r2,@object                      // @r2
	.section	.data.r2,"aw",@progbits
	.globl	r2
	.p2align	3, 0x0
r2:
	.xword	_ZL2s2
	.size	r2, 8

#--- a.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
_Z1Av, _Z1Bi, _Z1Ci

_Z1Av
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1

_Z1Bi
# Func Hash:
2222
# Num Counters:
1
# Counter Values:
1

_Z1Ci
# Func Hash:
3333
# Num Counters:
1
# Counter Values:
1

_Z1Di
# Func Hash:
4444
# Num Counters:
1
# Counter Values:
1

#--- a.orderfile
_Z1Av
_Z1Fi
_Z1Ei
_Z1Di
