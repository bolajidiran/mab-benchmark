#!/usr/bin/env bash
# OBD preprocessing (Fixes 1 to 5) and corrected S4 replay.
# Run from the mab-benchmark repository root:  bash obd_preprocessing/run_all.sh
set -euo pipefail
python obd_preprocessing/step1_fix5_integrity.py
python obd_preprocessing/step2_fix3_positions.py
python obd_preprocessing/step3_fix4_context.py
python obd_preprocessing/step4_fix1_fix2_slices.py
python obd_preprocessing/step5_run_s4_replay.py --jobs 2
# --- Chapter Four files (repository root) ---
python s4_obd_experiment.py
python s4_bernoulli_ts_experiment.py
# --- Sensitivity: keep the duplicate rows ---
python obd_preprocessing/step1_fix5_integrity.py --keep-duplicates
python obd_preprocessing/step2_fix3_positions.py --variant keep_duplicates
python obd_preprocessing/step3_fix4_context.py --variant keep_duplicates
python obd_preprocessing/step4_fix1_fix2_slices.py --variant keep_duplicates
python obd_preprocessing/step5_run_s4_replay.py --jobs 2 --variant keep_duplicates
python s4_obd_experiment.py --variant keep_duplicates
python s4_bernoulli_ts_experiment.py --variant keep_duplicates --skip-track0
python obd_preprocessing/compare_variants.py
