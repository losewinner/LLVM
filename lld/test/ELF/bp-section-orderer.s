# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: llvm-profdata merge a.proftext -o a.profdata

.ifdef GEN
#--- a.cc
const char s1[] = "hello world";
const char s2[] = "i am a string";
const char* r1 = s1;
const char** r2 = &r1;
void A() {
    return;
}

int B(int a) {
    A();
    return a + 1;
}

int C(int a) {
    A();
    return a + 2;
}

int D(int a) {
    return B(a + 2);
}

int E(int a) {
    return C(a + 2);
}

int F(int a) {
    return C(a + 3);
}

int main() {
    return 0;
}
#--- gen
echo '#--- a.s'
clang -target aarch64-linux-gnu -fdebug-compilation-dir='/proc/self/cwd' -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-asynchronous-unwind-tables -S -g a.cc -o -
.endif
#--- a.s
	.text
	.file	"a.cc"
	.file	0 "/proc/self/cwd" "a.cc" md5 0xd88df55d5eb7769f11cfb15e5857b68c
	.section	.text._Z1Av,"ax",@progbits
	.globl	_Z1Av                           // -- Begin function _Z1Av
	.p2align	2
	.type	_Z1Av,@function
_Z1Av:                                  // @_Z1Av
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
// %bb.0:
	.loc	0 6 5 prologue_end              // a.cc:6:5
	ret
.Ltmp0:
.Lfunc_end0:
	.size	_Z1Av, .Lfunc_end0-_Z1Av
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Bi,"ax",@progbits
	.globl	_Z1Bi                           // -- Begin function _Z1Bi
	.p2align	2
	.type	_Z1Bi,@function
_Z1Bi:                                  // @_Z1Bi
.Lfunc_begin1:
	.loc	0 9 0                           // a.cc:9:0
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
.Ltmp1:
	.loc	0 10 5 prologue_end             // a.cc:10:5
	bl	_Z1Av
	.loc	0 11 12                         // a.cc:11:12
	ldur	w8, [x29, #-4]
	.loc	0 11 14 is_stmt 0               // a.cc:11:14
	add	w0, w8, #1
	.loc	0 11 5 epilogue_begin           // a.cc:11:5
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Ltmp2:
.Lfunc_end1:
	.size	_Z1Bi, .Lfunc_end1-_Z1Bi
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Ci,"ax",@progbits
	.globl	_Z1Ci                           // -- Begin function _Z1Ci
	.p2align	2
	.type	_Z1Ci,@function
_Z1Ci:                                  // @_Z1Ci
.Lfunc_begin2:
	.loc	0 14 0 is_stmt 1                // a.cc:14:0
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
.Ltmp3:
	.loc	0 15 5 prologue_end             // a.cc:15:5
	bl	_Z1Av
	.loc	0 16 12                         // a.cc:16:12
	ldur	w8, [x29, #-4]
	.loc	0 16 14 is_stmt 0               // a.cc:16:14
	add	w0, w8, #2
	.loc	0 16 5 epilogue_begin           // a.cc:16:5
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Ltmp4:
.Lfunc_end2:
	.size	_Z1Ci, .Lfunc_end2-_Z1Ci
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Di,"ax",@progbits
	.globl	_Z1Di                           // -- Begin function _Z1Di
	.p2align	2
	.type	_Z1Di,@function
_Z1Di:                                  // @_Z1Di
.Lfunc_begin3:
	.loc	0 19 0 is_stmt 1                // a.cc:19:0
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
.Ltmp5:
	.loc	0 20 14 prologue_end            // a.cc:20:14
	ldur	w8, [x29, #-4]
	.loc	0 20 16 is_stmt 0               // a.cc:20:16
	add	w0, w8, #2
	.loc	0 20 12                         // a.cc:20:12
	bl	_Z1Bi
	.loc	0 20 5 epilogue_begin           // a.cc:20:5
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Ltmp6:
.Lfunc_end3:
	.size	_Z1Di, .Lfunc_end3-_Z1Di
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Ei,"ax",@progbits
	.globl	_Z1Ei                           // -- Begin function _Z1Ei
	.p2align	2
	.type	_Z1Ei,@function
_Z1Ei:                                  // @_Z1Ei
.Lfunc_begin4:
	.loc	0 23 0 is_stmt 1                // a.cc:23:0
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
.Ltmp7:
	.loc	0 24 14 prologue_end            // a.cc:24:14
	ldur	w8, [x29, #-4]
	.loc	0 24 16 is_stmt 0               // a.cc:24:16
	add	w0, w8, #2
	.loc	0 24 12                         // a.cc:24:12
	bl	_Z1Ci
	.loc	0 24 5 epilogue_begin           // a.cc:24:5
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Ltmp8:
.Lfunc_end4:
	.size	_Z1Ei, .Lfunc_end4-_Z1Ei
	.cfi_endproc
                                        // -- End function
	.section	.text._Z1Fi,"ax",@progbits
	.globl	_Z1Fi                           // -- Begin function _Z1Fi
	.p2align	2
	.type	_Z1Fi,@function
_Z1Fi:                                  // @_Z1Fi
.Lfunc_begin5:
	.loc	0 27 0 is_stmt 1                // a.cc:27:0
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	w0, [x29, #-4]
.Ltmp9:
	.loc	0 28 14 prologue_end            // a.cc:28:14
	ldur	w8, [x29, #-4]
	.loc	0 28 16 is_stmt 0               // a.cc:28:16
	add	w0, w8, #3
	.loc	0 28 12                         // a.cc:28:12
	bl	_Z1Ci
	.loc	0 28 5 epilogue_begin           // a.cc:28:5
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Ltmp10:
.Lfunc_end5:
	.size	_Z1Fi, .Lfunc_end5-_Z1Fi
	.cfi_endproc
                                        // -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
.Lfunc_begin6:
	.loc	0 31 0 is_stmt 1                // a.cc:31:0
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	mov	w0, wzr
	str	wzr, [sp, #12]
.Ltmp12:
	.loc	0 32 5 prologue_end epilogue_begin // a.cc:32:5
	add	sp, sp, #16
	ret
.Ltmp13:
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

	.type	r2,@object                      // @r2
	.section	.data.r2,"aw",@progbits
	.globl	r2
	.p2align	3, 0x0
r2:
	.xword	r1
	.size	r2, 8

	.section	.debug_abbrev,"",@progbits
	.byte	1                               // Abbreviation Code
	.byte	17                              // DW_TAG_compile_unit
	.byte	1                               // DW_CHILDREN_yes
	.byte	37                              // DW_AT_producer
	.byte	37                              // DW_FORM_strx1
	.byte	19                              // DW_AT_language
	.byte	5                               // DW_FORM_data2
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	114                             // DW_AT_str_offsets_base
	.byte	23                              // DW_FORM_sec_offset
	.byte	16                              // DW_AT_stmt_list
	.byte	23                              // DW_FORM_sec_offset
	.byte	27                              // DW_AT_comp_dir
	.byte	37                              // DW_FORM_strx1
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	85                              // DW_AT_ranges
	.byte	35                              // DW_FORM_rnglistx
	.byte	115                             // DW_AT_addr_base
	.byte	23                              // DW_FORM_sec_offset
	.byte	116                             // DW_AT_rnglists_base
	.byte	23                              // DW_FORM_sec_offset
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	2                               // Abbreviation Code
	.byte	52                              // DW_TAG_variable
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	3                               // Abbreviation Code
	.byte	15                              // DW_TAG_pointer_type
	.byte	0                               // DW_CHILDREN_no
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	4                               // Abbreviation Code
	.byte	38                              // DW_TAG_const_type
	.byte	0                               // DW_CHILDREN_no
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	5                               // Abbreviation Code
	.byte	36                              // DW_TAG_base_type
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	62                              // DW_AT_encoding
	.byte	11                              // DW_FORM_data1
	.byte	11                              // DW_AT_byte_size
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	6                               // Abbreviation Code
	.byte	52                              // DW_TAG_variable
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	110                             // DW_AT_linkage_name
	.byte	37                              // DW_FORM_strx1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	7                               // Abbreviation Code
	.byte	1                               // DW_TAG_array_type
	.byte	1                               // DW_CHILDREN_yes
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	8                               // Abbreviation Code
	.byte	33                              // DW_TAG_subrange_type
	.byte	0                               // DW_CHILDREN_no
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	55                              // DW_AT_count
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	9                               // Abbreviation Code
	.byte	36                              // DW_TAG_base_type
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	11                              // DW_AT_byte_size
	.byte	11                              // DW_FORM_data1
	.byte	62                              // DW_AT_encoding
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	10                              // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	0                               // DW_CHILDREN_no
	.byte	17                              // DW_AT_low_pc
	.byte	27                              // DW_FORM_addrx
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	110                             // DW_AT_linkage_name
	.byte	37                              // DW_FORM_strx1
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	11                              // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	1                               // DW_CHILDREN_yes
	.byte	17                              // DW_AT_low_pc
	.byte	27                              // DW_FORM_addrx
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	110                             // DW_AT_linkage_name
	.byte	37                              // DW_FORM_strx1
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	12                              // Abbreviation Code
	.byte	5                               // DW_TAG_formal_parameter
	.byte	0                               // DW_CHILDREN_no
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	13                              // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	0                               // DW_CHILDREN_no
	.byte	17                              // DW_AT_low_pc
	.byte	27                              // DW_FORM_addrx
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	0                               // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.hword	5                               // DWARF version number
	.byte	1                               // DWARF Unit Type
	.byte	8                               // Address Size (in bytes)
	.word	.debug_abbrev                   // Offset Into Abbrev. Section
	.byte	1                               // Abbrev [1] 0xc:0x110 DW_TAG_compile_unit
	.byte	0                               // DW_AT_producer
	.hword	4                               // DW_AT_language
	.byte	1                               // DW_AT_name
	.word	.Lstr_offsets_base0             // DW_AT_str_offsets_base
	.word	.Lline_table_start0             // DW_AT_stmt_list
	.byte	2                               // DW_AT_comp_dir
	.xword	0                               // DW_AT_low_pc
	.byte	0                               // DW_AT_ranges
	.word	.Laddr_table_base0              // DW_AT_addr_base
	.word	.Lrnglists_table_base0          // DW_AT_rnglists_base
	.byte	2                               // Abbrev [2] 0x2b:0xb DW_TAG_variable
	.byte	3                               // DW_AT_name
	.word	54                              // DW_AT_type
                                        // DW_AT_external
	.byte	0                               // DW_AT_decl_file
	.byte	3                               // DW_AT_decl_line
	.byte	2                               // DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               // Abbrev [3] 0x36:0x5 DW_TAG_pointer_type
	.word	59                              // DW_AT_type
	.byte	4                               // Abbrev [4] 0x3b:0x5 DW_TAG_const_type
	.word	64                              // DW_AT_type
	.byte	5                               // Abbrev [5] 0x40:0x4 DW_TAG_base_type
	.byte	4                               // DW_AT_name
	.byte	8                               // DW_AT_encoding
	.byte	1                               // DW_AT_byte_size
	.byte	2                               // Abbrev [2] 0x44:0xb DW_TAG_variable
	.byte	5                               // DW_AT_name
	.word	79                              // DW_AT_type
                                        // DW_AT_external
	.byte	0                               // DW_AT_decl_file
	.byte	4                               // DW_AT_decl_line
	.byte	2                               // DW_AT_location
	.byte	161
	.byte	1
	.byte	3                               // Abbrev [3] 0x4f:0x5 DW_TAG_pointer_type
	.word	54                              // DW_AT_type
	.byte	6                               // Abbrev [6] 0x54:0xc DW_TAG_variable
	.byte	6                               // DW_AT_name
	.word	96                              // DW_AT_type
	.byte	0                               // DW_AT_decl_file
	.byte	1                               // DW_AT_decl_line
	.byte	2                               // DW_AT_location
	.byte	161
	.byte	2
	.byte	8                               // DW_AT_linkage_name
	.byte	7                               // Abbrev [7] 0x60:0xc DW_TAG_array_type
	.word	59                              // DW_AT_type
	.byte	8                               // Abbrev [8] 0x65:0x6 DW_TAG_subrange_type
	.word	108                             // DW_AT_type
	.byte	12                              // DW_AT_count
	.byte	0                               // End Of Children Mark
	.byte	9                               // Abbrev [9] 0x6c:0x4 DW_TAG_base_type
	.byte	7                               // DW_AT_name
	.byte	8                               // DW_AT_byte_size
	.byte	7                               // DW_AT_encoding
	.byte	10                              // Abbrev [10] 0x70:0xc DW_TAG_subprogram
	.byte	3                               // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	111
	.byte	9                               // DW_AT_linkage_name
	.byte	10                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	5                               // DW_AT_decl_line
                                        // DW_AT_external
	.byte	11                              // Abbrev [11] 0x7c:0x1c DW_TAG_subprogram
	.byte	4                               // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin1       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	109
	.byte	11                              // DW_AT_linkage_name
	.byte	12                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	9                               // DW_AT_decl_line
	.word	279                             // DW_AT_type
                                        // DW_AT_external
	.byte	12                              // Abbrev [12] 0x8c:0xb DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	124
	.byte	23                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	9                               // DW_AT_decl_line
	.word	279                             // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	11                              // Abbrev [11] 0x98:0x1c DW_TAG_subprogram
	.byte	5                               // DW_AT_low_pc
	.word	.Lfunc_end2-.Lfunc_begin2       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	109
	.byte	14                              // DW_AT_linkage_name
	.byte	15                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	14                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
                                        // DW_AT_external
	.byte	12                              // Abbrev [12] 0xa8:0xb DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	124
	.byte	23                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	14                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	11                              // Abbrev [11] 0xb4:0x1c DW_TAG_subprogram
	.byte	6                               // DW_AT_low_pc
	.word	.Lfunc_end3-.Lfunc_begin3       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	109
	.byte	16                              // DW_AT_linkage_name
	.byte	17                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	19                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
                                        // DW_AT_external
	.byte	12                              // Abbrev [12] 0xc4:0xb DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	124
	.byte	23                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	19                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	11                              // Abbrev [11] 0xd0:0x1c DW_TAG_subprogram
	.byte	7                               // DW_AT_low_pc
	.word	.Lfunc_end4-.Lfunc_begin4       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	109
	.byte	18                              // DW_AT_linkage_name
	.byte	19                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	23                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
                                        // DW_AT_external
	.byte	12                              // Abbrev [12] 0xe0:0xb DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	124
	.byte	23                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	23                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	11                              // Abbrev [11] 0xec:0x1c DW_TAG_subprogram
	.byte	8                               // DW_AT_low_pc
	.word	.Lfunc_end5-.Lfunc_begin5       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	109
	.byte	20                              // DW_AT_linkage_name
	.byte	21                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	27                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
                                        // DW_AT_external
	.byte	12                              // Abbrev [12] 0xfc:0xb DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	124
	.byte	23                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	27                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	13                              // Abbrev [13] 0x108:0xf DW_TAG_subprogram
	.byte	9                               // DW_AT_low_pc
	.word	.Lfunc_end6-.Lfunc_begin6       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	111
	.byte	22                              // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	31                              // DW_AT_decl_line
	.word	279                             // DW_AT_type
                                        // DW_AT_external
	.byte	5                               // Abbrev [5] 0x117:0x4 DW_TAG_base_type
	.byte	13                              // DW_AT_name
	.byte	5                               // DW_AT_encoding
	.byte	4                               // DW_AT_byte_size
	.byte	0                               // End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.word	.Ldebug_list_header_end0-.Ldebug_list_header_start0 // Length
.Ldebug_list_header_start0:
	.hword	5                               // Version
	.byte	8                               // Address size
	.byte	0                               // Segment selector size
	.word	1                               // Offset entry count
.Lrnglists_table_base0:
	.word	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	3                               // DW_RLE_startx_length
	.byte	3                               //   start index
	.uleb128 .Lfunc_end0-.Lfunc_begin0      //   length
	.byte	3                               // DW_RLE_startx_length
	.byte	4                               //   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin1      //   length
	.byte	3                               // DW_RLE_startx_length
	.byte	5                               //   start index
	.uleb128 .Lfunc_end2-.Lfunc_begin2      //   length
	.byte	3                               // DW_RLE_startx_length
	.byte	6                               //   start index
	.uleb128 .Lfunc_end3-.Lfunc_begin3      //   length
	.byte	3                               // DW_RLE_startx_length
	.byte	7                               //   start index
	.uleb128 .Lfunc_end4-.Lfunc_begin4      //   length
	.byte	3                               // DW_RLE_startx_length
	.byte	8                               //   start index
	.uleb128 .Lfunc_end5-.Lfunc_begin5      //   length
	.byte	3                               // DW_RLE_startx_length
	.byte	9                               //   start index
	.uleb128 .Lfunc_end6-.Lfunc_begin6      //   length
	.byte	0                               // DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.word	100                             // Length of String Offsets Set
	.hword	5
	.hword	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               // string offset=0
.Linfo_string1:
	.asciz	"a.cc"                          // string offset=1
.Linfo_string2:
	.asciz	"/proc/self/cwd"                // string offset=6
.Linfo_string3:
	.asciz	"r1"                            // string offset=21
.Linfo_string4:
	.asciz	"char"                          // string offset=24
.Linfo_string5:
	.asciz	"r2"                            // string offset=29
.Linfo_string6:
	.asciz	"s1"                            // string offset=32
.Linfo_string7:
	.asciz	"__ARRAY_SIZE_TYPE__"           // string offset=35
.Linfo_string8:
	.asciz	"_ZL2s1"                        // string offset=55
.Linfo_string9:
	.asciz	"_Z1Av"                         // string offset=62
.Linfo_string10:
	.asciz	"A"                             // string offset=68
.Linfo_string11:
	.asciz	"_Z1Bi"                         // string offset=70
.Linfo_string12:
	.asciz	"B"                             // string offset=76
.Linfo_string13:
	.asciz	"int"                           // string offset=78
.Linfo_string14:
	.asciz	"_Z1Ci"                         // string offset=82
.Linfo_string15:
	.asciz	"C"                             // string offset=88
.Linfo_string16:
	.asciz	"_Z1Di"                         // string offset=90
.Linfo_string17:
	.asciz	"D"                             // string offset=96
.Linfo_string18:
	.asciz	"_Z1Ei"                         // string offset=98
.Linfo_string19:
	.asciz	"E"                             // string offset=104
.Linfo_string20:
	.asciz	"_Z1Fi"                         // string offset=106
.Linfo_string21:
	.asciz	"F"                             // string offset=112
.Linfo_string22:
	.asciz	"main"                          // string offset=114
.Linfo_string23:
	.asciz	"a"                             // string offset=119
	.section	.debug_str_offsets,"",@progbits
	.word	.Linfo_string0
	.word	.Linfo_string1
	.word	.Linfo_string2
	.word	.Linfo_string3
	.word	.Linfo_string4
	.word	.Linfo_string5
	.word	.Linfo_string6
	.word	.Linfo_string7
	.word	.Linfo_string8
	.word	.Linfo_string9
	.word	.Linfo_string10
	.word	.Linfo_string11
	.word	.Linfo_string12
	.word	.Linfo_string13
	.word	.Linfo_string14
	.word	.Linfo_string15
	.word	.Linfo_string16
	.word	.Linfo_string17
	.word	.Linfo_string18
	.word	.Linfo_string19
	.word	.Linfo_string20
	.word	.Linfo_string21
	.word	.Linfo_string22
	.word	.Linfo_string23
	.section	.debug_addr,"",@progbits
	.word	.Ldebug_addr_end0-.Ldebug_addr_start0 // Length of contribution
.Ldebug_addr_start0:
	.hword	5                               // DWARF version number
	.byte	8                               // Address size
	.byte	0                               // Segment selector size
.Laddr_table_base0:
	.xword	r1
	.xword	r2
	.xword	_ZL2s1
	.xword	.Lfunc_begin0
	.xword	.Lfunc_begin1
	.xword	.Lfunc_begin2
	.xword	.Lfunc_begin3
	.xword	.Lfunc_begin4
	.xword	.Lfunc_begin5
	.xword	.Lfunc_begin6
.Ldebug_addr_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z1Av
	.addrsig_sym _Z1Bi
	.addrsig_sym _Z1Ci
	.addrsig_sym _ZL2s1
	.addrsig_sym r1
	.section	.debug_line,"",@progbits
.Lline_table_start0:

# RUN: ld.lld -e main -o a.out a.o --irpgo-profile-sort=a.profdata --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=STARTUP
# RUN: ld.lld -e main -o a.out a.o --irpgo-profile-sort=a.profdata --verbose-bp-section-orderer --icf=all --compression-sort=none 2>&1 | FileCheck %s --check-prefix=STARTUP

# STARTUP: Ordered 3 sections using balanced partitioning

# RUN: ld.lld -e main -o - a.o --irpgo-profile-sort=a.profdata --symbol-ordering-file a.orderfile | llvm-nm --numeric-sort --format=just-symbols - | FileCheck %s --check-prefix=ORDERFILE

# ORDERFILE: _Z1Av
# ORDERFILE: _Z1Fi
# ORDERFILE: _Z1Ei
# ORDERFILE: _Z1Di
# ORDERFILE: _Z1Ci
# ORDERFILE: _Z1Bi
# ORDERFILE: main
# ORDERFILE: r1
# ORDERFILE: r2

# RUN: ld.lld -e main -o a.out a.o --verbose-bp-section-orderer --compression-sort=function 2>&1 | FileCheck %s --check-prefix=COMPRESSION-FUNC
# RUN: ld.lld -e main -o a.out a.o --verbose-bp-section-orderer --compression-sort=data 2>&1 | FileCheck %s --check-prefix=COMPRESSION-DATA
# RUN: ld.lld -e main -o a.out a.o --verbose-bp-section-orderer --compression-sort=both 2>&1 | FileCheck %s --check-prefix=COMPRESSION-BOTH
# RUN: ld.lld -e main -o a.out a.o --verbose-bp-section-orderer --compression-sort=both --irpgo-profile-sort=a.profdata 2>&1 | FileCheck %s --check-prefix=COMPRESSION-BOTH

# COMPRESSION-FUNC: Ordered 7 sections using balanced partitioning
# COMPRESSION-DATA: Ordered 3 sections using balanced partitioning
# COMPRESSION-BOTH: Ordered 10 sections using balanced partitioning

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