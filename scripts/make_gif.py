#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS = REPO_ROOT / "figs"
HIST = FIGS / "history"

NUM_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def sorted_frames(frames_dir: Path) -> list[Path]:
    files = list(frames_dir.glob("frame_*.*"))
    pairs = []
    for f in files:
        m = NUM_RE.match(f.name)
        if m:
            pairs.append((int(m.group(1)), f))
    pairs.sort(key=lambda x: x[0])
    return [p[1] for p in pairs]


def get_target_size(frames: list[Path]) -> tuple[int, int]:
    # Берём максимальный размер, чтобы ничего не обрезать
    max_w, max_h = 0, 0
    for p in frames:
        with Image.open(p) as im:
            w, h = im.size
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    return max_w, max_h


def normalize_frame(path: Path, target_size: tuple[int, int], bg_rgb=(255, 255, 255)) -> np.ndarray:
    """Приводим кадр к target_size и RGB.
    Если кадр меньше — центрируем и допаддим фоном.
    Если кадр больше — уменьшаем с сохранением пропорций и допаддим."""
    W, H = target_size

    im = Image.open(path).convert("RGBA")
    w, h = im.size

    # Если кадр больше целевого — уменьшаем
    if w > W or h > H:
        scale = min(W / w, H / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        im = im.resize((new_w, new_h), Image.LANCZOS)
        w, h = im.size

    canvas = Image.new("RGBA", (W, H), (*bg_rgb, 255))
    x = (W - w) // 2
    y = (H - h) // 2
    canvas.paste(im, (x, y), im)

    # GIF проще делать из RGB
    out = canvas.convert("RGB")
    return np.asarray(out)


def make_gif(frames_dir: Path, out_gif: Path, fps: float, every_k: int = 1, max_frames: int | None = None) -> None:
    frames = sorted_frames(frames_dir)
    if not frames:
        print(f"[skip] no frames found in {frames_dir}")
        return

    # прореживание/ограничение
    frames = frames[::every_k]
    if max_frames is not None:
        frames = frames[:max_frames]

    target_size = get_target_size(frames)
    duration = 1.0 / fps
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # потоковая запись — не делает np.stack на весь набор
    with imageio.get_writer(out_gif, mode="I", duration=duration, loop=0) as writer:
        for p in frames:
            arr = normalize_frame(p, target_size)
            writer.append_data(arr)

    print(f"[ok] {out_gif}  ({len(frames)} frames @ {fps} fps, size={target_size[0]}x{target_size[1]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--every_k", type=int, default=1, help="take every k-th frame (1=all)")
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    jobs = [
        # твои “исторические” анимации
        (HIST / "bm_anim",               HIST / "brownian.gif"),
        (HIST / "wiener_anim",           HIST / "wiener.gif"),
        (HIST / "qv_anim",               HIST / "ito_qv.gif"),
        (HIST / "bs_anim",               HIST / "black_scholes.gif"),
        (HIST / "diffusion_anim",        HIST / "diffusion.gif"),

        # “дополнительные” графики (если они у тебя есть и ты их генеришь)
        (HIST / "01_brownian_paths_anim", HIST / "01_brownian_paths.gif"),
        (HIST / "02_ou_paths_anim",       HIST / "02_ou_paths.gif"),
        (HIST / "03_gbm_paths_anim",      HIST / "03_gbm_paths.gif"),
        (HIST / "04_gbm_hist_anim",       HIST / "04_gbm_hist.gif"),
        (HIST / "forward1d_anim",         HIST / "forward_diffusion_1d.gif"),
        (HIST / "score_particles_anim",   HIST / "score_particles.gif"),
        (HIST / "em_refine_anim",         HIST / "em_refine.gif"),
    ]

    for frames_dir, out_gif in jobs:
        make_gif(frames_dir, out_gif, fps=args.fps, every_k=args.every_k, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
