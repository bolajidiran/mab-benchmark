#!/usr/bin/env bash
# Saito et al. validity check: all four tests. Run from the repository root.
set -e
D=validation/saito2020
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
python $D/s0_prepare_logs.py
python $D/test1_on_policy_ctr.py
python $D/test2_external_validity.py
python $D/test3_test4_ope_benchmark.py --all
python $D/summarise_validation.py
python $D/make_figures.py
