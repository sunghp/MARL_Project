"""정답 궤적에 YOLO 트랙 같은 잡음을 입힌다.

  calibrate  가시광 val의 YOLO 트랙(extract_tracks.py yolo 결과)에서 실제 잡음을 잰다.
             위치 흔들림·크기 흔들림(정답 크기 대비), 탐지 누락률, 끊김 빈도(초당), conf 분포 → noise.json
  apply      정답 궤적(extract_tracks.py gt 결과)에 그 잡음을 입혀 새 트랙 표를 만든다.

잡음은 정답 크기에 대한 비율로 재므로 해상도가 다른 적외선(320×256)에도 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASS_NAMES, load_table, save_table

Q_LEVELS = np.linspace(0, 1, 101)


def robust_std(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def calibrate(tracks: pd.DataFrame, gt_objects: pd.DataFrame) -> dict:
    m = tracks[(tracks["gt_obj"] >= 0) & (~tracks["is_gt"].astype(bool))].copy()
    if m.empty:
        raise ValueError("정답과 매칭된 YOLO 트랙 프레임이 없습니다. yolo 방식 결과를 넣었는지 확인하세요.")
    size = np.sqrt(m["gt_w"] * m["gt_h"]).clip(lower=1.0)
    m["ex"] = (m["cx"] - m["gt_cx"]) / size
    m["ey"] = (m["cy"] - m["gt_cy"]) / size
    m["lw"] = np.log(m["w"] / m["gt_w"])
    m["lh"] = np.log(m["h"] / m["gt_h"])
    m["cls_name"] = m["gt_cls"].astype(int).map(CLASS_NAMES)

    fps = tracks.groupby("clip_id")["fps"].first()
    go = gt_objects.copy()
    go["cls_name"] = go["gt_cls"].astype(int).map(CLASS_NAMES)
    go["dur_s"] = go["n_frames"] / go["clip_id"].map(fps).fillna(30.0)

    def stats(mm: pd.DataFrame, gg: pd.DataFrame) -> dict:
        return dict(
            n_frames=int(len(mm)),
            pos_sigma_x=robust_std(mm["ex"]), pos_sigma_y=robust_std(mm["ey"]),
            size_mu_w=float(mm["lw"].median()), size_sigma_w=robust_std(mm["lw"]),
            size_mu_h=float(mm["lh"].median()), size_sigma_h=robust_std(mm["lh"]),
            miss_rate=float(1 - gg["n_matched"].sum() / max(gg["n_frames"].sum(), 1)),
            break_rate_per_s=float((gg["n_tracks"] - 1).clip(lower=0).sum() / max(gg["dur_s"].sum(), 1e-9)),
            conf_quantiles=[float(v) for v in np.quantile(mm["conf"], Q_LEVELS)],
        )

    params = {"all": stats(m, go)}
    for name in CLASS_NAMES.values():
        mm, gg = m[m["cls_name"] == name], go[go["cls_name"] == name]
        if len(mm) >= 30:  # 표본이 너무 적으면 전체 값을 쓴다
            params[name] = stats(mm, gg)
    return {"classes": params}


def apply_noise(tracks: pd.DataFrame, noise: dict, draws=3, seed=0, break_gap=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for tid, g in tracks.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        p = noise["classes"].get(g["label"].iloc[0], noise["classes"]["all"])
        n = len(g)
        fps = float(g["fps"].iloc[0])
        for d in range(draws):
            h = g.copy()
            size = np.sqrt(h["w"] * h["h"]).clip(lower=1.0).to_numpy()
            h["cx"] = h["cx"] + rng.normal(0, p["pos_sigma_x"], n) * size
            h["cy"] = h["cy"] + rng.normal(0, p["pos_sigma_y"], n) * size
            h["w"] = h["w"] * np.exp(rng.normal(p["size_mu_w"], p["size_sigma_w"], n))
            h["h"] = h["h"] * np.exp(rng.normal(p["size_mu_h"], p["size_sigma_h"], n))
            h["conf"] = np.interp(rng.random(n), Q_LEVELS, p["conf_quantiles"])
            keep = rng.random(n) >= p["miss_rate"]
            # 끊김: 초당 빈도에 맞춰 자를 위치를 뽑고, 자른 뒤 몇 프레임은 비운다.
            n_breaks = rng.poisson(p["break_rate_per_s"] * n / fps) if n > 2 else 0
            cuts = np.sort(rng.choice(np.arange(1, n), size=min(n_breaks, n - 1), replace=False)) \
                if n_breaks else np.array([], dtype=int)
            for c in cuts:
                keep[c:c + break_gap] = False
            seg = np.searchsorted(cuts, np.arange(n), side="right")
            h["orig_track_id"] = tid
            h["draw"] = d
            h["track_id"] = [f"{tid}#d{d}s{s}" for s in seg]
            h = h[keep]
            h = h[h.groupby("track_id")["frame"].transform("size") >= 2]
            out.append(h)
    res = pd.concat(out, ignore_index=True) if out else tracks.iloc[0:0].copy()
    res["is_gt"] = False
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--tracks", required=True, help="yolo 방식 tracks.parquet")
    c.add_argument("--gt-objects", required=True, help="같은 실행의 gt_objects.csv")
    c.add_argument("--out", required=True, help="noise.json")
    a = sub.add_parser("apply")
    a.add_argument("--tracks", required=True, help="gt 방식 tracks.parquet")
    a.add_argument("--noise", required=True)
    a.add_argument("--out", required=True, help="출력 폴더")
    a.add_argument("--draws", type=int, default=3, help="정답 트랙 하나당 잡음 버전 수")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--break-gap", type=int, default=3, help="끊긴 뒤 비우는 프레임 수")
    args = p.parse_args(argv)

    if args.cmd == "calibrate":
        noise = calibrate(load_table(args.tracks), pd.read_csv(args.gt_objects))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(noise, indent=2), encoding="utf-8")
        for k, v in noise["classes"].items():
            print(f"{k:6s} 프레임 {v['n_frames']:6d} | 위치 σ {v['pos_sigma_x']:.3f}/{v['pos_sigma_y']:.3f} "
                  f"| 누락 {v['miss_rate']:.2f} | 끊김 {v['break_rate_per_s']:.3f}/s")
    else:
        noise = json.loads(Path(args.noise).read_text(encoding="utf-8"))
        res = apply_noise(load_table(args.tracks), noise, args.draws, args.seed, args.break_gap)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = save_table(res, out / "tracks")
        print(f"저장: {path} (트랙 {res['track_id'].nunique()}개, 행 {len(res)}개)")


if __name__ == "__main__":
    main()
