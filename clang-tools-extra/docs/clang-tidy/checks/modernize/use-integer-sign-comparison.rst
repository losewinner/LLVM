.. title:: clang-tidy - modernize-use-integer-sign-comparison

modernize-use-integer-sign-comparison
=====================================

Replace comparisons between signed and unsigned integers with their safe
``std::cmp_*`` alternative.

Examples of fixes created by the check:

.. code-block:: c++

    uint func(int bla)
    {
        uint result;
        if (result == bla)
            return 0;

        return 1;
    }

becomes

.. code-block:: c++

    #include <utility>

    uint func(int bla)
    {
        uint result;
        if (std::cmp_equal(result, bla))
            return 0;

        return 1;
    }
