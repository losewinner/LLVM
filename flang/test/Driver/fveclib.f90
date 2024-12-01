! RUN: %flang -### -c -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB %s
! RUN: %flang -### -c -fveclib=Accelerate %s 2>&1 | FileCheck -check-prefix CHECK-ACCELERATE %s
! RUN: %flang -### -c -fveclib=libmvec %s 2>&1 | FileCheck -check-prefix CHECK-libmvec %s
! RUN: %flang -### -c -fveclib=MASSV %s 2>&1 | FileCheck -check-prefix CHECK-MASSV %s
! RUN: %flang -### -c -fveclib=Darwin_libsystem_m %s 2>&1 | FileCheck -check-prefix CHECK-DARWIN_LIBSYSTEM_M %s
! RUN: %flang -### -c --target=aarch64-none-none -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-SLEEF %s
! RUN: %flang -### -c --target=aarch64-none-none -fveclib=ArmPL %s 2>&1 | FileCheck -check-prefix CHECK-ARMPL %s
! RUN: %flang -### -c --target=aarch64-apple-darwin -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB-DARWIN %s
! RUN: not %flang -c -fveclib=something %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

! CHECK-NOLIB: "-fveclib=none"
! CHECK-ACCELERATE: "-fveclib=Accelerate"
! CHECK-libmvec: "-fveclib=libmvec"
! CHECK-MASSV: "-fveclib=MASSV"
! CHECK-DARWIN_LIBSYSTEM_M: "-fveclib=Darwin_libsystem_m"
! CHECK-SLEEF: "-fveclib=SLEEF"
! CHECK-ARMPL: "-fveclib=ArmPL"
! CHECK-NOLIB-DARWIN: "-fveclib=none"

! CHECK-INVALID: error: invalid value 'something' in '-fveclib=something'

! RUN: not %flang --target=x86-none-none -c -fveclib=SLEEF %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=x86-none-none -c -fveclib=ArmPL %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=aarch64-none-none -c -fveclib=LIBMVEC-X86 %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! RUN: not %flang --target=aarch64-none-none -c -fveclib=SVML %s 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
! CHECK-ERROR: unsupported option {{.*}} for target

! RUN: %flang -fveclib=Accelerate %s -target arm64-apple-ios8.0.0 -### 2>&1 | FileCheck --check-prefix=CHECK-LINK %s
! CHECK-LINK: "-framework" "Accelerate"

! TODO: if we add support for -nostdlib or -nodefaultlibs we need to test that
! these prevent "-framework Accelerate" being added on Darwin

! RUN: %flang -### --target=aarch64-pc-windows-msvc -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL-MSVC %s
! RUN: %flang -### --target=aarch64-linux-gnu -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL %s
! RUN: %flang -### --target=arm64-apple-darwin -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-LINKING-ARMPL %s
! CHECK-LINKING-ARMPL: "-lamath"
! CHECK-LINKING-ARMPL-SAME: "-lm"
! CHECK-LINKING-ARMPL-MSVC: "--dependent-lib=amath"

! RUN: %flang -### --target=aarch64-linux-gnu -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_arch_subdir -frtlib-add-rpath -fveclib=ArmPL %s 2>&1 | FileCheck --check-prefix=CHECK-RPATH-ARMPL %s
! CHECK-RPATH-ARMPL: "-lamath"
! CHECK-RPATH-ARMPL-SAME: "-lm"
! We need to see "-rpath" at least twice, one for veclib, one for the Fortran runtime
! CHECK-RPATH-ARMPL-SAME: "-rpath"
! CHECK-RPATH-ARMPL-SAME: "-rpath"
