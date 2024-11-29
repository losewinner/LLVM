.. title:: clang-tidy - modernize-use-span-first-last

modernize-use-span-first-last
============================

Checks for uses of ``std::span::subspan()`` that can be replaced with clearer
``first()`` or ``last()`` member functions.

Covered scenarios:

==================================== ==================================
Expression                           Replacement
------------------------------------ ----------------------------------
``s.subspan(0, n)``                  ``s.first(n)``
``s.subspan(n)``                     ``s.last(s.size() - n)``
==================================== ==================================

Non-zero offset with count (like ``subspan(1, n)``) has no direct equivalent
using ``first()`` or ``last()``, so these cases are not transformed.