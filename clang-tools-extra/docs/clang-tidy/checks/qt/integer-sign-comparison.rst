.. title:: clang-tidy - qt-integer-sign-comparison

qt-integer-sign-comparison
=============================

The qt-integer-sign-comparison check is an alias, please see
:doc:`modernize-use-integer-sign-comparison <../modernize/use-integer-sign-comparison>`
for more information.

Examples of fixes created by the check:

.. code-block:: c++

  uint func(int a, uint b) {
    return a == b;
  }

in C++17 becomes

.. code-block:: c++

  <QtCore/q20utility.h>

  uint func(int a, uint b) {
    return (q20::cmp_equal(result, bla))
  }

in C++20 becomes

.. code-block:: c++

  #include <utility>

  uint func(int a, uint b) {
    return (std::cmp_equal(result, bla))
  }


Options
-------

.. option:: IncludeStyle

  A string specifying which include-style is used, `llvm` or `google`.
  Default is `llvm`.

.. option:: IsQtApplication

  When `true`, it is assumed that the code being analyzed is using the Qt framework.
  Default is `false`.
