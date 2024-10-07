!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine test1
!ERROR: OMPX_BARE clause is only allowed on combined TARGET TEAMS
  !$omp target ompx_bare
  !$omp end target
end

subroutine test2
  !$omp target
!ERROR: OMPX_BARE clause is only allowed on combined TARGET TEAMS
  !$omp teams ompx_bare
  !$omp end teams
  !$omp end target
end

subroutine test3
!No errors
  !$omp target teams ompx_bare
  !$omp end target teams
end
