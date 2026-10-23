"""궤적 추출 공통 함수: 클래스 대응, 라벨·프레임 읽기, GT 연결, 트랙-GT 매칭, 트랙 잇기, 배경 이동."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# 팀 클래스 대응표와 같은 번호를 쓴다.
CLASS_NAMES = {0: "bird", 1: "drone", 2: "other"}
NONE = "none"            # 정답과 매칭되지 않은 트랙 = YOLO 오탐 트랙
AMBIGUOUS = "ambiguous"  # 매칭 비율이 애매해 학습에서 뺄 트랙

# CSV 라벨의 원본 이름 → 클래스 번호. 표에 없는 이름이 나오면 오류로 멈춘다.
NAME_TO_ID = {
    "bird": 0, "drone": 1, "uav": 1,
    "airplane": 2, "aeroplane": 2, "helicopter": 2, "other": 2,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- 클립 목록

def pair_id(clip_id: str) -> str:
    """가시광(V_)과 적외선(IR_) 쌍을 묶는 키. 예: V_DRONE_001, IR_DRONE_001 → DRONE_001."""
    return re.sub(r"^(V|IR)_", "", clip_id.strip(), flags=re.IGNORECASE)


def read_list(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]


def check_not_eval(clips: list[str], eval_ids_path: str | None) -> None:
    """평가 클립(또는 그 쌍)이 하나라도 있으면 종료코드 1로 멈춘다."""
    if not eval_ids_path:
        return
    eval_pairs = {pair_id(c) for c in read_list(eval_ids_path)}
    bad = sorted(c for c in clips if pair_id(c) in eval_pairs)
    if bad:
        print(f"[중단] 평가 클립이 입력에 들어 있습니다 ({len(bad)}개): {bad[:5]} ...", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------- 프레임

class FrameSource:
    """영상 파일(video_dir/clip+ext) 또는 이미지 폴더(frames_dir/clip/)에서 프레임을 순서대로 읽는다."""

    def __init__(self, clip_id: str, video_dir=None, frames_dir=None, video_ext=".mp4", fps=None):
        self.clip_id = clip_id
        self.video = None
        self.images: list[Path] = []
        if video_dir and (Path(video_dir) / f"{clip_id}{video_ext}").is_file():
            self.video = Path(video_dir) / f"{clip_id}{video_ext}"
        elif frames_dir and (Path(frames_dir) / clip_id).is_dir():
            self.images = sorted(p for p in (Path(frames_dir) / clip_id).iterdir()
                                 if p.suffix.lower() in IMAGE_EXTS)
        self.fps = fps

    @property
    def exists(self) -> bool:
        return self.video is not None or bool(self.images)

    def __iter__(self):
        import cv2
        if self.video is not None:
            cap = cv2.VideoCapture(str(self.video))
            if not cap.isOpened():
                raise RuntimeError(f"영상을 열 수 없습니다: {self.video}")
            meta_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps is None and meta_fps and meta_fps > 0:
                self.fps = float(meta_fps)
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
                i += 1
            cap.release()
        else:
            for i, p in enumerate(self.images):
                frame = cv2.imread(str(p))
                if frame is None:
                    raise RuntimeError(f"이미지를 읽을 수 없습니다: {p}")
                yield i, frame


class BackgroundMotion:
    """이웃 프레임 사이 배경 이동(px)을 위상 상관으로 추정한다. 카메라가 오른쪽으로 돌면 dx는 음수."""

    def __init__(self, scale: float = 0.5):
        self.scale = scale
        self.prev = None
        self.window = None

    def __call__(self, frame) -> tuple[float, float, float]:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cur = small.astype(np.float32)
        if self.prev is None or self.prev.shape != cur.shape:
            self.prev = cur
            self.window = cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
            return 0.0, 0.0, 0.0
        (sx, sy), resp = cv2.phaseCorrelate(self.prev, cur, self.window)
        self.prev = cur
        return sx / self.scale, sy / self.scale, float(resp)


# ---------------------------------------------------------------- 정답 라벨

def _last_int(name: str) -> int | None:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def load_gt_yolo(clip_id, labels_dir, layout, img_w, img_h, frame_offset=0) -> pd.DataFrame:
    """프레임별 YOLO txt(정규화 cx cy w h) → px 단위 표. 프레임 번호는 파일 이름의 마지막 숫자."""
    root = Path(labels_dir)
    if layout == "subdir":
        files = sorted((root / clip_id).glob("*.txt"))
    else:  # flat: {clip}_{frame}.txt
        pat = re.compile(rf"^{re.escape(clip_id)}_(\d+)\.txt$")
        files = sorted(p for p in root.glob(f"{clip_id}_*.txt") if pat.match(p.name))
    if not files:
        raise FileNotFoundError(f"{clip_id}: 라벨 파일이 없습니다 ({root}, layout={layout})")
    rows = []
    for p in files:
        frame = _last_int(p.stem) - frame_offset
        for line in p.read_text().split("\n"):
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            if cls not in CLASS_NAMES:
                raise ValueError(f"{p}: 대응표에 없는 클래스 번호 {cls}")
            cx, cy, w, h = map(float, parts[1:5])
            rows.append((frame, cls, cx * img_w, cy * img_h, w * img_w, h * img_h))
    return pd.DataFrame(rows, columns=["frame", "cls", "cx", "cy", "w", "h"])


def load_gt_csv(clip_id, csv_df: pd.DataFrame, frame_offset=0) -> pd.DataFrame:
    """CSV(clip, frame, cls, x, y, w, h; 왼쪽 위 기준 px) → 같은 표."""
    sub = csv_df[csv_df["clip"] == clip_id]
    if sub.empty:
        raise FileNotFoundError(f"{clip_id}: CSV에 라벨이 없습니다")
    names = sub["cls"].astype(str).str.lower().str.strip()
    unknown = sorted(set(names) - set(NAME_TO_ID))
    if unknown:
        raise ValueError(f"{clip_id}: 대응표에 없는 라벨 {unknown}")
    return pd.DataFrame({
        "frame": sub["frame"].astype(int) - frame_offset,
        "cls": names.map(NAME_TO_ID).astype(int),
        "cx": sub["x"] + sub["w"] / 2, "cy": sub["y"] + sub["h"] / 2,
        "w": sub["w"].astype(float), "h": sub["h"].astype(float),
    }).reset_index(drop=True)


def link_gt(gt: pd.DataFrame, max_gap=10, dist_min=20.0, dist_rel=2.0) -> pd.DataFrame:
    """정답 박스를 프레임 사이로 이어 물체 번호(gt_obj)를 붙인다. 같은 클래스끼리만, 등속 예측 위치 기준."""
    gt = gt.sort_values("frame").reset_index(drop=True)
    gt["gt_obj"] = -1
    objs: list[dict] = []  # last_frame, pos, vel, size, cls
    for frame, idx in gt.groupby("frame").groups.items():
        idx = list(idx)
        live = [k for k, o in enumerate(objs) if 0 < frame - o["last"] <= max_gap]
        assigned = {}
        if live:
            cost = np.full((len(idx), len(live)), 1e9)
            for a, i in enumerate(idx):
                r = gt.loc[i]
                for b, k in enumerate(live):
                    o = objs[k]
                    if o["cls"] != r.cls:
                        continue
                    pred = o["pos"] + o["vel"] * (frame - o["last"])
                    d = float(np.hypot(r.cx - pred[0], r.cy - pred[1]))
                    if d <= max(dist_min, dist_rel * o["size"]):
                        cost[a, b] = d
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e9:
                    assigned[idx[a]] = live[b]
        for i in idx:
            r = gt.loc[i]
            pos = np.array([r.cx, r.cy])
            size = max(r.w, r.h)
            if i in assigned:
                o = objs[assigned[i]]
                o["vel"] = (pos - o["pos"]) / (frame - o["last"])
                o.update(last=frame, pos=pos, size=size)
                gt.at[i, "gt_obj"] = assigned[i]
            else:
                objs.append(dict(last=frame, pos=pos, vel=np.zeros(2), size=size, cls=r.cls))
                gt.at[i, "gt_obj"] = len(objs) - 1
    return gt


# ---------------------------------------------------------------- 트랙 ↔ 정답

def match_frame(pred_xy: np.ndarray, gt_xy: np.ndarray, gt_size: np.ndarray,
                min_px=5.0, rel=1.0) -> list[tuple[int, int]]:
    """한 프레임에서 예측 중심과 정답 중심을 1:1로 매칭한다. 거리 ≤ max(min_px, rel×정답 크기)."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return []
    d = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    gate = np.maximum(min_px, rel * gt_size)[None, :]
    cost = np.where(d <= gate, d, 1e9)
    return [(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e9]


def label_tracks(rows: pd.DataFrame, pos_ratio=0.5, neg_ratio=0.1, purity=0.8) -> pd.DataFrame:
    """트랙마다 라벨을 정한다.
    매칭 비율 ≥ pos_ratio 이고 가장 많이 매칭된 정답 물체 비중 ≥ purity → 그 물체의 클래스
    매칭 비율 ≤ neg_ratio → none (오탐 트랙)
    그 밖 → ambiguous (학습에서 뺀다)
    """
    out = []
    for tid, g in rows.groupby("track_id", sort=False):
        n = len(g)
        m = g[g["gt_obj"] >= 0]
        ratio = len(m) / n
        if len(m):
            counts = m["gt_obj"].value_counts()
            top_obj, top_n = int(counts.index[0]), int(counts.iloc[0])
            top_cls = int(m.loc[m["gt_obj"] == top_obj, "gt_cls"].iloc[0])
            pur = top_n / len(m)
        else:
            top_obj, top_cls, pur = -1, -1, 0.0
        if ratio >= pos_ratio and pur >= purity:
            label = CLASS_NAMES[top_cls]
        elif ratio <= neg_ratio:
            label = NONE
        else:
            label = AMBIGUOUS
        out.append(dict(track_id=tid, n_frames=n, matched_ratio=ratio, purity=pur,
                        gt_obj_major=top_obj, label=label))
    return pd.DataFrame(out)


def stitch_tracks(rows: pd.DataFrame, max_gap=15, dist_min=10.0, dist_rel=3.0) -> pd.DataFrame:
    """끊긴 트랙을 잇는다(과거만 사용). 새 트랙이 시작될 때, 그 전에 끝난 트랙의 예측 위치 근처면 같은 번호로 바꾼다.
    작고 빠른 물체는 IoU가 0이 되어 트래커가 자주 끊기므로 쓴다. 실시간에서도 같은 규칙으로 돌릴 수 있다.
    """
    rows = rows.sort_values(["track_id", "frame"]).copy()
    info = []
    for tid, g in rows.groupby("track_id", sort=False):
        xy = g[["cx", "cy"]].to_numpy()
        fr = g["frame"].to_numpy()
        k = min(5, len(g))
        vel = (xy[-1] - xy[-k]) / max(fr[-1] - fr[-k], 1) if k > 1 else np.zeros(2)
        info.append(dict(tid=tid, start=fr[0], end=fr[-1], first=xy[0], last=xy[-1], vel=vel,
                         size=float(np.median(np.maximum(g["w"], g["h"])))))
    info.sort(key=lambda d: (d["start"], str(d["tid"])))
    root = {}
    tails: list[dict] = []  # 이어 붙일 수 있는 체인의 끝
    for t in info:
        best, best_d = None, None
        for c in tails:
            gap = t["start"] - c["end"]
            if not 0 < gap <= max_gap:
                continue
            pred = c["last"] + c["vel"] * gap
            d = float(np.linalg.norm(t["first"] - pred))
            if d <= max(dist_min, dist_rel * c["size"]) and (best_d is None or d < best_d):
                best, best_d = c, d
        if best is not None:
            root[t["tid"]] = best["root"]
            tails.remove(best)
        else:
            root[t["tid"]] = t["tid"]
        tails.append(dict(t, root=root[t["tid"]]))
    rows["track_id"] = rows["track_id"].map(root)
    # 두 트랙이 같은 프레임을 가지면 conf가 높은 쪽만 남긴다.
    rows = rows.sort_values("conf", ascending=False).drop_duplicates(["track_id", "frame"])
    return rows.sort_values(["track_id", "frame"]).reset_index(drop=True)


# ---------------------------------------------------------------- 저장

def save_table(df: pd.DataFrame, path_stem: Path) -> Path:
    """parquet로 저장하고, pyarrow가 없으면 csv로 저장한다."""
    try:
        path = path_stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except ImportError:
        path = path_stem.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
