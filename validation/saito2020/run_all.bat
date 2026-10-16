@echo off
rem Saito et al. validity check: all four tests. Run from the repository root.
set D=validation\saito2020
if "%OMP_NUM_THREADS%"=="" set OMP_NUM_THREADS=1
python %D%\s0_prepare_logs.py || exit /b 1
python %D%\test1_on_policy_ctr.py || exit /b 1
python %D%\test2_external_validity.py || exit /b 1
python %D%\test3_test4_ope_benchmark.py --all || exit /b 1
python %D%\summarise_validation.py || exit /b 1
python %D%\make_figures.py || exit /b 1
