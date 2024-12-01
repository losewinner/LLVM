! RUN: %flang -### --target=aarch64-windows-msvc -resource-dir=%S/Inputs/resource_dir %S/Inputs/hello.f90 -v 2>&1 | FileCheck %s --check-prefixes=MSVC
! RUN: %flang -### --target=aarch64-windows-msvc -resource-dir=%S/Inputs/resource_dir -fms-runtime-lib=static_dbg %S/Inputs/hello.f90 -v 2>&1 | FileCheck %s --check-prefixes=MSVC-DEBUG
! RUN: %flang -### --target=aarch64-windows-msvc -resource-dir=%S/Inputs/resource_dir -fms-runtime-lib=dll %S/Inputs/hello.f90 -v 2>&1 | FileCheck %s --check-prefixes=MSVC-DLL
! RUN: %flang -### --target=aarch64-windows-msvc -resource-dir=%S/Inputs/resource_dir -fms-runtime-lib=dll_dbg %S/Inputs/hello.f90 -v 2>&1 | FileCheck %s --check-prefixes=MSVC-DLL-DEBUG

! MSVC: -fc1
! MSVC-SAME: -D_MT
! MSVC-SAME: --dependent-lib=libcmt
! MSVC-SAME: --dependent-lib=flang_rt.static.lib

! MSVC-DEBUG: -fc1
! MSVC-DEBUG-SAME: -D_MT
! MSVC-DEBUG-SAME: -D_DEBUG
! MSVC-DEBUG-SAME: --dependent-lib=libcmtd
! MSVC-DEBUG-SAME: --dependent-lib=flang_rt.static_dbg.lib

! MSVC-DLL: -fc1
! MSVC-DLL-SAME: -D_MT
! MSVC-DLL-SAME: -D_DLL
! MSVC-DLL-SAME: --dependent-lib=msvcrt
! MSVC-DLL-SAME: --dependent-lib=flang_rt.dynamic.lib

! MSVC-DLL-DEBUG: -fc1
! MSVC-DLL-DEBUG-SAME: -D_MT
! MSVC-DLL-DEBUG-SAME: -D_DEBUG
! MSVC-DLL-DEBUG-SAME: -D_DLL
! MSVC-DLL-DEBUG-SAME: --dependent-lib=msvcrtd
! MSVC-DLL-DEBUG-SAME: --dependent-lib=flang_rt.dynamic_dbg.lib
