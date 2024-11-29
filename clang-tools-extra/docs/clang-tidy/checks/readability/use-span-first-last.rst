.. title:: clang-tidy - readability-use-span-first-last

readability-use-span-first-last
=============================

Checks for uses of ``std::span::subspan()`` that can be replaced with clearer
``first()`` or ``last()`` member functions. These dedicated methods were added 
to C++20 to provide more expressive alternatives to common subspan operations.

Covered scenarios:

==================================== ==================================
Expression                           Replacement
------------------------------------ ----------------------------------
``s.subspan(0, n)``                  ``s.first(n)``
``s.subspan(s.size() - n)``          ``s.last(n)``
==================================== ==================================

Non-zero offset with count (like ``subspan(1, n)``) or offset-only calls 
(like ``subspan(n)``) have no clearer equivalent using ``first()`` or 
``last()``, so these cases are not transformed.

This check is only active when C++20 or later is used.