"""Write kuairec_dataset/small_matrix_compact.npz (user_id, video_id, watch_ratio)
from KuaiRec's small_matrix.csv: 25 MB instead of 406 MB, loaded in a second."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "kuairec_dataset")
csv = next(iter(sorted(root.rglob("small_matrix.csv"))))
df = pd.read_csv(csv, usecols=["user_id", "video_id", "watch_ratio"])
np.savez_compressed(root / "small_matrix_compact.npz", user_id=df.user_id.to_numpy(np.int32),
                    video_id=df.video_id.to_numpy(np.int32), watch_ratio=df.watch_ratio.to_numpy(np.float32))
print(len(df), "rows ->", root / "small_matrix_compact.npz")
