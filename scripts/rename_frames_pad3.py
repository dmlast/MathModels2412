import re
from pathlib import Path

d = Path("figs/history/bm_anim")
for p in d.glob("frame_*.png"):
    m = re.search(r"frame_(\d+)\.png$", p.name)
    if not m:
        continue
    i = int(m.group(1))
    p.rename(d / f"frame_{i:03d}.png")
print("Done")
