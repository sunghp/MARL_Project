"""svanstrom 영상에서 궤적(트랙) 데이터를 뽑는다.

두 가지 방식:
  yolo  YOLO → 트래커 → 정답과 매칭해 트랙 라벨(bird/drone/other/none/ambiguous). 가시광 val 클립용.
  gt    정답 박스를 이어 정답 궤적을 만든다. 적외선 클립, 가시광 train 클립용(이후 noise.py로 잡음을 입힌다).

출력(--out 폴더):
  tracks.parquet       프레임마다 한 행(트랙 하나의 한 프레임)
  track_summary.csv    트랙마다 한 행(길이, 매칭 비율, 라벨)
  gt_objects.csv       정답 물체마다 한 행(프레임 수, 매칭된 프레임 수, 걸친 트랙 수) — 잡음 보정에 쓴다
  clip_summary.csv     클립마다 한 행(트랙 수, 끊김, 카메라 확인용 지표)
  run_config.json      실행 인자
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CLASS_NAMES, FrameSource, BackgroundMotion, check_not_eval, label_tracks,
                    link_gt, load_gt_csv, load_gt_yolo, match_frame, pair_id, read_list,
                    save_table, stitch_tracks)

TRACK_COLS = ["source", "clip_id", "pair_id", "track_id", "frame", "t_sec", "fps", "img_w", "img_h",
              "cx", "cy", "w", "h", "conf", "cls_yolo",
              "gt_obj", "gt_cls", "gt_cx", "gt_cy", "gt_w", "gt_h",
              "bg_dx", "bg_dy", "bg_resp", "is_gt", "label"]


class UltralyticsTracker:
    """Ultralytics YOLO + 내장 트래커(ByteTrack/BoT-SORT). 프레임 하나를 받아 트랙 박스를 돌려준다."""

    def __init__(self, weights, tracker_cfg, conf=0.01, imgsz=416, device=None):
        self.weights, self.tracker_cfg = weights, tracker_cfg
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.model = None

    def reset(self):
        # 클립마다 새로 불러 트래커 상태를 비운다(nano 가중치라 빠르다).
        from ultralytics import YOLO
        self.model = YOLO(self.weights)

    def __call__(self, frame) -> np.ndarray:
        """반환: (k, 7) = track_id, cx, cy, w, h, conf, cls"""
        r = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz,
                             tracker=self.tracker_cfg, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None or len(b) == 0:
            return np.zeros((0, 7))
        return np.column_stack([b.id.cpu().numpy(), b.xywh.cpu().numpy(),
                                b.conf.cpu().numpy(), b.cls.cpu().numpy()])


def scan_frames(src: FrameSource, tracker=None, bg_motion=True):
    """프레임을 한 번 읽으며 배경 이동과 (yolo 방식이면) 트랙 박스를 모은다."""
    bgm = BackgroundMotion() if bg_motion else None
    bg_rows, det_rows = [], []
    img_w = img_h = None
    n = 0
    for i, frame in src:
        if img_w is None:
            img_h, img_w = frame.shape[:2]
        dx, dy, resp = bgm(frame) if bgm else (0.0, 0.0, 0.0)
        bg_rows.append((i, dx, dy, resp))
        if tracker is not None:
            for t in tracker(frame):
                det_rows.append((i, *t))
        n = i + 1
    bg = pd.DataFrame(bg_rows, columns=["frame", "bg_dx", "bg_dy", "bg_resp"])
    dets = pd.DataFrame(det_rows, columns=["frame", "tid", "cx", "cy", "w", "h", "conf", "cls_yolo"])
    return n, img_w, img_h, bg, dets


def load_gt(clip, args, img_w, img_h, csv_df):
    if args.gt_csv:
        gt = load_gt_csv(clip, csv_df, args.frame_offset)
    else:
        gt = load_gt_yolo(clip, args.labels_dir, args.label_layout, img_w, img_h, args.frame_offset)
    return link_gt(gt, args.link_max_gap, args.link_dist_min, args.link_dist_rel)


def match_tracks(dets: pd.DataFrame, gt: pd.DataFrame, args) -> pd.DataFrame:
    """프레임마다 트랙 박스와 정답 박스를 매칭해 gt_* 열을 채운다."""
    dets = dets.copy()
    for c in ["gt_obj", "gt_cls"]:
        dets[c] = -1
    for c in ["gt_cx", "gt_cy", "gt_w", "gt_h"]:
        dets[c] = np.nan
    gt_by_frame = {f: g for f, g in gt.groupby("frame")}
    for f, d in dets.groupby("frame"):
        g = gt_by_frame.get(f)
        if g is None:
            continue
        pairs = match_frame(d[["cx", "cy"]].to_numpy(), g[["cx", "cy"]].to_numpy(),
                            np.maximum(g["w"], g["h"]).to_numpy(), args.match_min_px, args.match_rel)
        for i, j in pairs:
            di, gj = d.index[i], g.iloc[j]
            dets.loc[di, ["gt_obj", "gt_cls"]] = [int(gj.gt_obj), int(gj.cls)]
            dets.loc[di, ["gt_cx", "gt_cy", "gt_w", "gt_h"]] = [gj.cx, gj.cy, gj.w, gj.h]
    return dets


def gt_object_table(gt: pd.DataFrame, rows: pd.DataFrame | None) -> pd.DataFrame:
    out = []
    for obj, g in gt.groupby("gt_obj"):
        rec = dict(gt_obj=int(obj), gt_cls=int(g["cls"].iloc[0]), n_frames=len(g),
                   first_frame=int(g["frame"].min()), last_frame=int(g["frame"].max()),
                   size_px=float(np.median(np.maximum(g["w"], g["h"]))))
        if rows is not None:
            m = rows[rows["gt_obj"] == obj]
            rec.update(n_matched=int(m["frame"].nunique()), n_tracks=int(m["track_id"].nunique()))
        out.append(rec)
    return pd.DataFrame(out)


def process_clip(clip, args, tracker, csv_df):
    src = FrameSource(clip, args.video_dir, args.frames_dir, args.video_ext, args.fps)
    if src.exists:
        if tracker is not None:
            tracker.reset()
        n, img_w, img_h, bg, dets = scan_frames(src, tracker, not args.no_bg_motion)
        fps = src.fps or 30.0
    else:
        if tracker is not None:
            raise FileNotFoundError(f"{clip}: 영상/프레임을 찾을 수 없습니다")
        if not args.img_size:
            raise FileNotFoundError(f"{clip}: 영상이 없으면 --img-size W H 가 필요합니다")
        n = None
        img_w, img_h = args.img_size
        fps = args.fps or 30.0
        bg = pd.DataFrame({"frame": pd.Series(dtype="int64"), "bg_dx": pd.Series(dtype=float),
                           "bg_dy": pd.Series(dtype=float), "bg_resp": pd.Series(dtype=float)})
        dets = None

    gt = load_gt(clip, args, img_w, img_h, csv_df)
    mismatch = bool(n is not None and gt["frame"].max() >= n)

    if tracker is None:  # 정답 궤적
        rows = gt.rename(columns={"cls": "gt_cls"}).copy()
        rows["track_id"] = [f"{clip}:gt{o}" for o in rows["gt_obj"]]
        rows["conf"], rows["cls_yolo"], rows["is_gt"] = 1.0, -1, True
        for c in ["cx", "cy", "w", "h"]:
            rows[f"gt_{c}"] = rows[c]
        rows["label"] = rows["gt_cls"].map(CLASS_NAMES)
        summary = rows.groupby("track_id").agg(n_frames=("frame", "size"), label=("label", "first"))
        summary = summary.reset_index().assign(matched_ratio=1.0, purity=1.0)
        gto = gt_object_table(gt, None)
    else:  # YOLO 트랙
        dets["track_id"] = [f"{clip}:{int(t)}" for t in dets["tid"]]
        dets = dets.drop(columns="tid")
        if args.stitch and len(dets):
            dets = stitch_tracks(dets, max_gap=round(args.stitch_gap_s * fps),
                                 dist_min=args.stitch_dist_min, dist_rel=args.stitch_dist_rel)
        rows = match_tracks(dets, gt, args) if len(dets) else dets.assign(gt_obj=[], gt_cls=[])
        rows["is_gt"] = False
        summary = label_tracks(rows, args.pos_ratio, args.neg_ratio, args.purity) if len(rows) else \
            pd.DataFrame(columns=["track_id", "n_frames", "matched_ratio", "purity", "gt_obj_major", "label"])
        rows = rows.merge(summary[["track_id", "label"]], on="track_id", how="left")
        gto = gt_object_table(gt, rows)

    rows = rows.merge(bg, on="frame", how="left")
    rows[["bg_dx", "bg_dy", "bg_resp"]] = rows[["bg_dx", "bg_dy", "bg_resp"]].astype(float).fillna(0.0)
    rows = rows.assign(source=args.source, clip_id=clip, pair_id=pair_id(clip), fps=fps,
                       img_w=img_w, img_h=img_h, t_sec=rows["frame"] / fps)
    for c in TRACK_COLS:
        if c not in rows:
            rows[c] = np.nan
    rows = rows[TRACK_COLS].sort_values(["track_id", "frame"]).reset_index(drop=True)

    summary = summary.assign(clip_id=clip, source=args.source)
    gto = gto.assign(clip_id=clip, source=args.source)

    # 카메라 확인: 정답 중심이 화면 가운데(가로·세로 ±15%)에 있는 비율. 높으면 카메라가 물체를 따라간 영상일 가능성.
    center = ((gt["cx"] / img_w - 0.5).abs() < 0.15) & ((gt["cy"] / img_h - 0.5).abs() < 0.15)
    labels = summary["label"].value_counts().to_dict()
    clip_info = dict(clip_id=clip, source=args.source, n_frames=n, fps=fps, img_w=img_w, img_h=img_h,
                     n_gt_boxes=len(gt), n_gt_objects=gt["gt_obj"].nunique(), n_tracks=len(summary),
                     gt_center_ratio=float(center.mean()),
                     bg_motion_px=float(np.hypot(bg["bg_dx"].astype(float), bg["bg_dy"].astype(float)).mean())
                     if len(bg) else np.nan,
                     tracks_per_gt_object=float(gto["n_tracks"].mean()) if "n_tracks" in gto else np.nan,
                     label_beyond_video=mismatch,
                     **{f"n_{k}": labels.get(k, 0) for k in ["bird", "drone", "other", "none", "ambiguous"]})
    return rows, summary, gto, clip_info


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ["yolo", "gt"]:
        s = sub.add_parser(mode)
        s.add_argument("--clips", required=True, help="클립 ID 목록 txt (한 줄에 하나)")
        s.add_argument("--source", required=True, choices=["vis", "ir"])
        s.add_argument("--out", required=True)
        s.add_argument("--video-dir", help="영상 폴더 (클립ID + --video-ext)")
        s.add_argument("--video-ext", default=".mp4")
        s.add_argument("--frames-dir", help="이미지 폴더 (frames-dir/클립ID/*.jpg)")
        s.add_argument("--labels-dir", help="YOLO txt 라벨 폴더")
        s.add_argument("--label-layout", choices=["subdir", "flat"], default="subdir",
                       help="subdir: labels/클립ID/*.txt, flat: labels/클립ID_프레임.txt")
        s.add_argument("--gt-csv", help="YOLO txt 대신 CSV 라벨(clip,frame,cls,x,y,w,h)")
        s.add_argument("--frame-offset", type=int, default=0, help="라벨 파일 번호가 1부터면 1")
        s.add_argument("--img-size", type=int, nargs=2, metavar=("W", "H"), help="영상 없이 돌릴 때 이미지 크기")
        s.add_argument("--fps", type=float, help="영상 메타데이터가 없을 때 fps")
        s.add_argument("--eval-ids", help="평가 클립 ID 목록. 입력과 겹치면(쌍 포함) 종료코드 1")
        s.add_argument("--no-bg-motion", action="store_true", help="배경 이동 계산 생략")
        s.add_argument("--link-max-gap", type=int, default=10, help="정답 연결: 최대 빈 프레임")
        s.add_argument("--link-dist-min", type=float, default=20.0)
        s.add_argument("--link-dist-rel", type=float, default=2.0)
        if mode == "yolo":
            s.add_argument("--weights", required=True)
            s.add_argument("--tracker", default=str(Path(__file__).parent / "trackers" / "bytetrack_small.yaml"))
            s.add_argument("--conf", type=float, default=0.01, help="낮게 둬야 오탐 트랙(none)이 생긴다")
            s.add_argument("--imgsz", type=int, default=416)
            s.add_argument("--device", default=None)
            s.add_argument("--match-min-px", type=float, default=5.0, help="매칭 거리 하한(px)")
            s.add_argument("--match-rel", type=float, default=1.0, help="매칭 거리 = 정답 크기 × 이 값")
            s.add_argument("--pos-ratio", type=float, default=0.5)
            s.add_argument("--neg-ratio", type=float, default=0.1)
            s.add_argument("--purity", type=float, default=0.8)
            s.add_argument("--stitch", action="store_true", help="끊긴 트랙 잇기")
            s.add_argument("--stitch-gap-s", type=float, default=0.5)
            s.add_argument("--stitch-dist-min", type=float, default=10.0)
            s.add_argument("--stitch-dist-rel", type=float, default=3.0)
    return p


def main(argv=None, tracker=None):
    args = build_parser().parse_args(argv)
    if not args.gt_csv and not args.labels_dir:
        sys.exit("--labels-dir 또는 --gt-csv 가 필요합니다")
    clips = read_list(args.clips)
    check_not_eval(clips, args.eval_ids)
    if args.mode == "yolo" and tracker is None:
        tracker = UltralyticsTracker(args.weights, args.tracker, args.conf, args.imgsz, args.device)
    if args.mode == "gt":
        tracker = None
    csv_df = pd.read_csv(args.gt_csv) if args.gt_csv else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, all_sum, all_gto, infos, errors = [], [], [], [], []
    t0 = time.time()
    for k, clip in enumerate(clips, 1):
        try:
            rows, summ, gto, info = process_clip(clip, args, tracker, csv_df)
        except Exception as e:  # 한 클립이 실패해도 나머지는 돌리고, 끝에 종료코드 1
            errors.append((clip, repr(e)))
            print(f"[{k}/{len(clips)}] {clip} 실패: {e}", file=sys.stderr)
            continue
        all_rows.append(rows); all_sum.append(summ); all_gto.append(gto); infos.append(info)
        print(f"[{k}/{len(clips)}] {clip}: 트랙 {info['n_tracks']}개, 정답 물체 {info['n_gt_objects']}개 "
              f"({time.time() - t0:.0f}s)")

    if all_rows:
        tracks_path = save_table(pd.concat(all_rows, ignore_index=True), out / "tracks")
        pd.concat(all_sum, ignore_index=True).to_csv(out / "track_summary.csv", index=False)
        pd.concat(all_gto, ignore_index=True).to_csv(out / "gt_objects.csv", index=False)
        clip_df = pd.DataFrame(infos)
        clip_df.to_csv(out / "clip_summary.csv", index=False)
        summ = pd.concat(all_sum, ignore_index=True)
        print(f"\n저장: {tracks_path}")
        print("트랙 라벨별 개수:", summ["label"].value_counts().to_dict())
        print(f"정답 중심이 화면 가운데에 있는 비율(클립 평균): {clip_df['gt_center_ratio'].mean():.2f}")
        if clip_df["label_beyond_video"].any():
            print("[경고] 라벨 프레임 번호가 영상 길이를 넘는 클립:",
                  clip_df.loc[clip_df["label_beyond_video"], "clip_id"].tolist())
    cfg = {k: v for k, v in vars(args).items()}
    cfg["errors"] = errors
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        print(f"\n[실패] {len(errors)}개 클립: {[c for c, _ in errors]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
