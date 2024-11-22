; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure dxil operation function calls for length are generated for half/float.

define noundef half @test_length_half2(<2 x half> noundef %p0) {
; CHECK-LABEL: define noundef half @test_length_half2(
; CHECK-SAME: <2 x half> noundef [[P0:%.*]]) {
; CHECK:  [[ENTRY:.*:]]
; CHECK:    [[MUL_I:%.*]] = fmul <2 x half> [[P0]], [[P0]]
; CHECK:    [[TMP0:%.*]] = extractelement <2 x half> [[MUL_I]], i64 0
; CHECK:    [[TMP1:%.*]] = fadd half 0xH0000, [[TMP0]]
; CHECK:    [[TMP2:%.*]] = extractelement <2 x half> [[MUL_I]], i64 1
; CHECK:    [[TMP3:%.*]] = fadd half [[TMP1]], [[TMP2]]
; EXPCHECK: [[HLSL_LENGTH:%.*]] = call half @llvm.sqrt.f16(half [[TMP3]])
; DOPCHECK: [[HLSL_LENGTH:%.*]] = call half @dx.op.unary.f16(i32 24, half [[TMP3]])
; CHECK:    ret half [[HLSL_LENGTH]]
;
entry:

  %mul.i = fmul <2 x half> %p0, %p0
  %rdx.fadd.i = call half @llvm.vector.reduce.fadd.v2f16(half 0xH0000, <2 x half> %mul.i)
  %hlsl.length = call half @llvm.sqrt.f16(half %rdx.fadd.i)
  ret half %hlsl.length
}

define noundef half @test_length_half3(<3 x half> noundef %p0) {
; CHECK-LABEL: define noundef half @test_length_half3(
; CHECK-SAME: <3 x half> noundef [[P0:%.*]]) {
; CHECK:  [[ENTRY:.*:]]
; CHECK:    [[MUL_I:%.*]] = fmul <3 x half> [[P0]], [[P0]]
; CHECK:    [[TMP0:%.*]] = extractelement <3 x half> [[MUL_I]], i64 0
; CHECK:    [[TMP1:%.*]] = fadd half 0xH0000, [[TMP0]]
; CHECK:    [[TMP2:%.*]] = extractelement <3 x half> [[MUL_I]], i64 1
; CHECK:    [[TMP3:%.*]] = fadd half [[TMP1]], [[TMP2]]
; CHECK:    [[TMP4:%.*]] = extractelement <3 x half> [[MUL_I]], i64 2
; CHECK:    [[TMP5:%.*]] = fadd half [[TMP3]], [[TMP4]]
; EXPCHECK: [[HLSL_LENGTH:%.*]] = call half @llvm.sqrt.f16(half [[TMP5]])
; DOPCHECK: [[HLSL_LENGTH:%.*]] = call half @dx.op.unary.f16(i32 24, half [[TMP5]])
; CHECK:    ret half [[HLSL_LENGTH]]
;
entry:

  %mul.i = fmul <3 x half> %p0, %p0
  %rdx.fadd.i = call half @llvm.vector.reduce.fadd.v2f16(half 0xH0000, <3 x half> %mul.i)
  %hlsl.length = call half @llvm.sqrt.f16(half %rdx.fadd.i)
  ret half %hlsl.length
}

define noundef half @test_length_half4(<4 x half> noundef %p0) {
; CHECK-LABEL: define noundef half @test_length_half4(
; CHECK-SAME: <4 x half> noundef [[P0:%.*]]) {
; CHECK:  [[ENTRY:.*:]]
; CHECK:    [[MUL_I:%.*]] = fmul <4 x half> [[P0]], [[P0]]
; CHECK:    [[TMP0:%.*]] = extractelement <4 x half> [[MUL_I]], i64 0
; CHECK:    [[TMP1:%.*]] = fadd half 0xH0000, [[TMP0]]
; CHECK:    [[TMP2:%.*]] = extractelement <4 x half> [[MUL_I]], i64 1
; CHECK:    [[TMP3:%.*]] = fadd half [[TMP1]], [[TMP2]]
; CHECK:    [[TMP4:%.*]] = extractelement <4 x half> [[MUL_I]], i64 2
; CHECK:    [[TMP5:%.*]] = fadd half [[TMP3]], [[TMP4]]
; CHECK:    [[TMP6:%.*]] = extractelement <4 x half> [[MUL_I]], i64 3
; CHECK:    [[TMP7:%.*]] = fadd half [[TMP5]], [[TMP6]]
; EXPCHECK: [[HLSL_LENGTH:%.*]] = call half @llvm.sqrt.f16(half [[TMP7]])
; DOPCHECK: [[HLSL_LENGTH:%.*]] = call half @dx.op.unary.f16(i32 24, half [[TMP7]])
; CHECK:    ret half [[HLSL_LENGTH]]
;
entry:

  %mul.i = fmul <4 x half> %p0, %p0
  %rdx.fadd.i = call half @llvm.vector.reduce.fadd.v2f16(half 0xH0000, <4 x half> %mul.i)
  %hlsl.length = call half @llvm.sqrt.f16(half %rdx.fadd.i)
  ret half %hlsl.length
}

define noundef float @test_length_float2(<2 x float> noundef %p0) {
; CHECK-LABEL: define noundef float @test_length_float2(
; CHECK-SAME: <2 x float> noundef [[P0:%.*]]) {
; CHECK:  [[ENTRY:.*:]]
; CHECK:    [[MUL_I:%.*]] = fmul <2 x float> [[P0]], [[P0]]
; CHECK:    [[TMP0:%.*]] = extractelement <2 x float> [[MUL_I]], i64 0
; CHECK:    [[TMP1:%.*]] = fadd float 0.000000e+00, [[TMP0]]
; CHECK:    [[TMP2:%.*]] = extractelement <2 x float> [[MUL_I]], i64 1
; CHECK:    [[TMP3:%.*]] = fadd float [[TMP1]], [[TMP2]]
; EXPCHECK: [[HLSL_LENGTH:%.*]] = call float @llvm.sqrt.f32(float [[TMP3]])
; DOPCHECK: [[HLSL_LENGTH:%.*]] = call float @dx.op.unary.f32(i32 24, float [[TMP3]])
; CHECK:    ret float [[HLSL_LENGTH]]
;
entry:

  %mul.i = fmul <2 x float> %p0, %p0
  %rdx.fadd.i = call float @llvm.vector.reduce.fadd.v2f32(float 0.000000e+00, <2 x float> %mul.i)
  %hlsl.length = call float @llvm.sqrt.f32(float %rdx.fadd.i)
  ret float %hlsl.length
}

define noundef float @test_length_float3(<3 x float> noundef %p0) {
; CHECK-LABEL: define noundef float @test_length_float3(
; CHECK-SAME: <3 x float> noundef [[P0:%.*]]) {
; CHECK:  [[ENTRY:.*:]]
; CHECK:    [[MUL_I:%.*]] = fmul <3 x float> [[P0]], [[P0]]
; CHECK:    [[TMP0:%.*]] = extractelement <3 x float> [[MUL_I]], i64 0
; CHECK:    [[TMP1:%.*]] = fadd float 0.000000e+00, [[TMP0]]
; CHECK:    [[TMP2:%.*]] = extractelement <3 x float> [[MUL_I]], i64 1
; CHECK:    [[TMP3:%.*]] = fadd float [[TMP1]], [[TMP2]]
; CHECK:    [[TMP4:%.*]] = extractelement <3 x float> [[MUL_I]], i64 2
; CHECK:    [[TMP5:%.*]] = fadd float [[TMP3]], [[TMP4]]
; EXPCHECK: [[HLSL_LENGTH:%.*]] = call float @llvm.sqrt.f32(float [[TMP5]])
; DOPCHECK: [[HLSL_LENGTH:%.*]] = call float @dx.op.unary.f32(i32 24, float [[TMP5]])
; CHECK:    ret float [[HLSL_LENGTH]]
;
entry:

  %mul.i = fmul <3 x float> %p0, %p0
  %rdx.fadd.i = call float @llvm.vector.reduce.fadd.v2f32(float 0.000000e+00, <3 x float> %mul.i)
  %hlsl.length = call float @llvm.sqrt.f32(float %rdx.fadd.i)
  ret float %hlsl.length
}

define noundef float @test_length_float4(<4 x float> noundef %p0) {
; CHECK-LABEL: define noundef float @test_length_float4(
; CHECK-SAME: <4 x float> noundef [[P0:%.*]]) {
; CHECK:  [[ENTRY:.*:]]
; CHECK:    [[MUL_I:%.*]] = fmul <4 x float> [[P0]], [[P0]]
; CHECK:    [[TMP0:%.*]] = extractelement <4 x float> [[MUL_I]], i64 0
; CHECK:    [[TMP1:%.*]] = fadd float 0.000000e+00, [[TMP0]]
; CHECK:    [[TMP2:%.*]] = extractelement <4 x float> [[MUL_I]], i64 1
; CHECK:    [[TMP3:%.*]] = fadd float [[TMP1]], [[TMP2]]
; CHECK:    [[TMP4:%.*]] = extractelement <4 x float> [[MUL_I]], i64 2
; CHECK:    [[TMP5:%.*]] = fadd float [[TMP3]], [[TMP4]]
; CHECK:    [[TMP6:%.*]] = extractelement <4 x float> [[MUL_I]], i64 3
; CHECK:    [[TMP7:%.*]] = fadd float [[TMP5]], [[TMP6]]
; EXPCHECK: [[HLSL_LENGTH:%.*]] = call float @llvm.sqrt.f32(float [[TMP7]])
; DOPCHECK: [[HLSL_LENGTH:%.*]] = call float @dx.op.unary.f32(i32 24, float [[TMP7]])
; CHECK:    ret float [[HLSL_LENGTH]]
;
entry:

  %mul.i = fmul <4 x float> %p0, %p0
  %rdx.fadd.i = call float @llvm.vector.reduce.fadd.v2f32(float 0.000000e+00, <4 x float> %mul.i)
  %hlsl.length = call float @llvm.sqrt.f32(float %rdx.fadd.i)
  ret float %hlsl.length
}
