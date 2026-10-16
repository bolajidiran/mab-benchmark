@echo off
REM OBD preprocessing (Fixes 1 to 5) and corrected S4 replay.
REM Main analysis, Chapter Four files, then the keep-duplicates sensitivity.
REM Run from the mab-benchmark repository root:
REM     obd_preprocessing\run_all.bat
python obd_preprocessing\step1_fix5_integrity.py || exit /b 1
python obd_preprocessing\step2_fix3_positions.py || exit /b 1
python obd_preprocessing\step3_fix4_context.py   || exit /b 1
python obd_preprocessing\step4_fix1_fix2_slices.py || exit /b 1
python obd_preprocessing\step5_run_s4_replay.py --jobs 2 || exit /b 1
REM --- Chapter Four files (repository root) ---
python s4_obd_experiment.py || exit /b 1
python s4_bernoulli_ts_experiment.py || exit /b 1
REM --- Sensitivity: keep the duplicate rows ---
python obd_preprocessing\step1_fix5_integrity.py --keep-duplicates || exit /b 1
python obd_preprocessing\step2_fix3_positions.py --variant keep_duplicates || exit /b 1
python obd_preprocessing\step3_fix4_context.py --variant keep_duplicates || exit /b 1
python obd_preprocessing\step4_fix1_fix2_slices.py --variant keep_duplicates || exit /b 1
python obd_preprocessing\step5_run_s4_replay.py --jobs 2 --variant keep_duplicates || exit /b 1
python s4_obd_experiment.py --variant keep_duplicates || exit /b 1
python s4_bernoulli_ts_experiment.py --variant keep_duplicates --skip-track0 || exit /b 1
python obd_preprocessing\compare_variants.py || exit /b 1
