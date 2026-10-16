# Datasets

**Open Bandit Dataset (S4 Tier 2).** Saito et al. (2021), ZOZOTOWN, released with the
Open Bandit Pipeline at <https://github.com/st-tech/zr-obp>. Download the full
dataset, place it in `open_bandit_dataset/`, and run the preprocessing steps in
`obd_preprocessing/` (`run_all.sh` or `run_all.bat`). The data are not redistributed.

**KuaiRec (S4 Tier 1, bias-free).** C. Gao et al. (2022), CIKM, licence CC BY-SA 4.0.
Download `KuaiRec.zip` from <https://zenodo.org/records/18164998> into
`kuairec_dataset/`, then run `python experiments/kuairec_compact.py` once to write the
25 MB `small_matrix_compact.npz` that `S4_kuairec` reads. The environment draws 80
videos (seed 2026) observed for at least 99% of users and defines a click as a watch
ratio of at least 2.0. The data are not redistributed.

**Synthetic settings** need no data.
