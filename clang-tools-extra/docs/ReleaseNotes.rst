====================================================
Extra Clang Tools |release| |ReleaseNotesTitle|
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Extra Clang Tools |version| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release |release|. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Extra Clang Tools |release|?
==========================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

- The ``clang-pseudo`` tool is incomplete and does not have active maintainers,
  so it has been removed. See
  `the RFC <https://discourse.llvm.org/t/removing-pseudo-parser/71131/>`_ for
  more details.

...

Improvements to clangd
----------------------

Inlay hints
^^^^^^^^^^^

Diagnostics
^^^^^^^^^^^

Semantic Highlighting
^^^^^^^^^^^^^^^^^^^^^

Compile flags
^^^^^^^^^^^^^

Hover
^^^^^

Code completion
^^^^^^^^^^^^^^^

Code actions
^^^^^^^^^^^^

Signature help
^^^^^^^^^^^^^^

Cross-references
^^^^^^^^^^^^^^^^

Objective-C
^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

Improvements to clang-doc
-------------------------

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- Improved :program:`run-clang-tidy.py` script. Fixed minor shutdown noise
  happening on certain platforms when interrupting the script.

New checks
^^^^^^^^^^


- New :doc:`boost-use-ranges
  <clang-tidy/checks/boost/use-ranges>` check.

  Detects calls to standard library iterator algorithms that could be replaced
  with a Boost ranges version instead.

- New :doc:`bugprone-crtp-constructor-accessibility
  <clang-tidy/checks/bugprone/crtp-constructor-accessibility>` check.

  Detects error-prone Curiously Recurring Template Pattern usage, when the CRTP
  can be constructed outside itself and the derived class.

- New :doc:`bugprone-pointer-arithmetic-on-polymorphic-object
  <clang-tidy/checks/bugprone/pointer-arithmetic-on-polymorphic-object>` check.

  Finds pointer arithmetic performed on classes that contain a virtual function.

- New :doc:`bugprone-return-const-ref-from-parameter
  <clang-tidy/checks/bugprone/return-const-ref-from-parameter>` check.

  Detects return statements that return a constant reference parameter as constant
  reference. This may cause use-after-free errors if the caller uses xvalues as
  arguments.

- New :doc:`bugprone-suspicious-stringview-data-usage
  <clang-tidy/checks/bugprone/suspicious-stringview-data-usage>` check.

  Identifies suspicious usages of ``std::string_view::data()`` that could lead
  to reading out-of-bounds data due to inadequate or incorrect string null
  termination.

- New :doc:`misc-use-internal-linkage
  <clang-tidy/checks/misc/use-internal-linkage>` check.

  Detects variables and functions that can be marked as static or moved into
  an anonymous namespace to enforce internal linkage.

- New :doc:`modernize-min-max-use-initializer-list
  <clang-tidy/checks/modernize/min-max-use-initializer-list>` check.

  Replaces nested ``std::min`` and ``std::max`` calls with an initializer list
  where applicable.

- New :doc:`modernize-use-cpp-style-comments
  <clang-tidy/checks/modernize/use-cpp-style-comments>` check.

  Detects C Style comments and suggests to use C++ style comments instead.

- New :doc:`modernize-use-designated-initializers
  <clang-tidy/checks/modernize/use-designated-initializers>` check.

  Finds initializer lists for aggregate types that could be
  written as designated initializers instead.

- New :doc:`modernize-use-ranges
  <clang-tidy/checks/modernize/use-ranges>` check.

  Detects calls to standard library iterator algorithms that could be replaced
  with a ranges version instead.

- New :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check.

  Converts calls to ``absl::StrFormat``, or other functions via
  configuration options, to C++20's ``std::format``, or another function
  via a configuration option, modifying the format string appropriately and
  removing now-unnecessary calls to ``std::string::c_str()`` and
  ``std::string::data()``.

- New :doc:`readability-enum-initial-value
  <clang-tidy/checks/readability/enum-initial-value>` check.

  Enforces consistent style for enumerators' initialization, covering three
  styles: none, first only, or all initialized explicitly.

- New :doc:`readability-math-missing-parentheses
  <clang-tidy/checks/readability/math-missing-parentheses>` check.

  Check for missing parentheses in mathematical expressions that involve
  operators of different priorities.

- New :doc:`readability-use-std-min-max
  <clang-tidy/checks/readability/use-std-min-max>` check.

  Replaces certain conditional statements with equivalent calls to
  ``std::min`` or ``std::max``.


New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-arr39-c <clang-tidy/checks/cert/arr39-c>` to
  :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone/sizeof-expression>` was added.

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Improved :doc:`bugprone-casting-through-void
  <clang-tidy/checks/bugprone/casting-through-void>` check to suggest replacing
  the offending code with ``reinterpret_cast``, to more clearly express intent.

- Improved :doc:`bugprone-dangling-handle
  <clang-tidy/checks/bugprone/dangling-handle>` check to treat `std::span` as a
  handle class.

- Improved :doc:`bugprone-forwarding-reference-overload
  <clang-tidy/checks/bugprone/forwarding-reference-overload>` check by fixing
  a crash when determining if an ``enable_if[_t]`` was found.

- Improved :doc:`bugprone-posix-return
  <clang-tidy/checks/bugprone/posix-return>` check to support integer literals
  as LHS and posix call as RHS of comparison.

- Improved :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone/sizeof-expression>` check to find suspicious
  usages of ``sizeof()``, ``alignof()``, and ``offsetof()`` when adding or
  subtracting from a pointer.

- Improved :doc:`bugprone-unchecked-optional-access
  <clang-tidy/checks/bugprone/unchecked-optional-access>` to support
  `bsl::optional` and `bdlb::NullableValue` from
  <https://github.com/bloomberg/bde>_.

- Improved :doc:`bugprone-unsafe-functions
  <clang-tidy/checks/bugprone/unsafe-functions>` check to allow specifying
  additional functions to match.

- Improved :doc:`cert-flp30-c <clang-tidy/checks/cert/flp30-c>` check to
  fix false positive that floating point variable is only used in increment
  expression.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>` check to
  avoid false positive when member initialization depends on a structured
  binding variable.

- Improved :doc:`misc-definitions-in-headers
  <clang-tidy/checks/misc/definitions-in-headers>` check by rewording the
  diagnostic note that suggests adding ``inline``.

- Improved :doc:`modernize-avoid-c-arrays
  <clang-tidy/checks/modernize/avoid-c-arrays>` check to suggest using ``std::span``
  as a replacement for parameters of incomplete C array type in C++20 and 
  ``std::array`` or ``std::vector`` before C++20.

- Improved :doc:`modernize-use-std-format
  <clang-tidy/checks/modernize/use-std-format>` check to support replacing
  member function calls too.

- Improved :doc:`misc-unconventional-assign-operator
  <clang-tidy/checks/misc/unconventional-assign-operator>` check to avoid
  false positive for C++23 deducing this.

- Improved :doc:`modernize-min-max-use-initializer-list
  <clang-tidy/checks/modernize/min-max-use-initializer-list>` check by fixing
  a false positive when only an implicit conversion happened inside an
  initializer list.

- Improved :doc:`modernize-use-nullptr
  <clang-tidy/checks/modernize/use-nullptr>` check to also recognize
  ``NULL``/``__null`` (but not ``0``) when used with a templated type.

- Improved :doc:`modernize-use-std-print
  <clang-tidy/checks/modernize/use-std-print>` check to support replacing
  member function calls too.

- Improved :doc:`readability-enum-initial-value
  <clang-tidy/checks/readability/enum-initial-value>` check by only issuing
  diagnostics for the definition of an ``enum``, and by fixing a typo in the
  diagnostic.

- Improved :doc:`performance-avoid-endl
  <clang-tidy/checks/performance/avoid-endl>` check to use ``std::endl`` as
  placeholder when lexer cannot get source text.

- Improved :doc:`readability-container-contains
  <clang-tidy/checks/readability/container-contains>` check to let it work on
  any class that has a ``contains`` method.

- Improved :doc:`readability-implicit-bool-conversion
  <clang-tidy/checks/readability/implicit-bool-conversion>` check
  by adding the option `UseUpperCaseLiteralSuffix` to select the
  case of the literal suffix in fixes.

- Improved :doc:`readability-redundant-smartptr-get
  <clang-tidy/checks/readability/redundant-smartptr-get>` check to
  remove `->`, when redundant `get()` is removed.

Removed checks
^^^^^^^^^^^^^^

Miscellaneous
^^^^^^^^^^^^^

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to clang-include-fixer
-----------------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...

Improvements to pp-trace
------------------------

Clang-tidy Visual Studio plugin
-------------------------------
