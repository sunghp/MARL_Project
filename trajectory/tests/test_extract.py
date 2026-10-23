"""합성 데이터로 궤적 추출 전체 흐름을 확인한다. 실행: python -m pytest trajectory/tests -q"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_tracks  # noqa: E402
import noise  # noqa: E402
from common import BackgroundMotion, link_gt, stitch_tracks  # noqa: E402

W, H, N = 160, 128, 60


def obj_pos(i):
    return 20 + 2 * i, 40 + 10 * np.sin(i / 5)  # 오른쪽으로 날며 위아래로 흔들림


def make_clip(root: Path, clip: str, cls=0):
    rng = np.random.default_rng(0)
    bg = (rng.random((H, W)) * 120).astype(np.uint8)
    (root / "frames" / clip).mkdir(parents=True)
    (root / "labels" / clip).mkdir(parents=True)
    for i in range(N):
        img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        cx, cy = obj_pos(i)
        cv2.rectangle(img, (int(cx - 4), int(cy - 3)), (int(cx + 4), int(cy + 3)), (255, 255, 255), -1)
        cv2.imwrite(str(root / "frames" / clip / f"{i:06d}.png"), img)
        (root / "labels" / clip / f"{i:06d}.txt").write_text(f"{cls} {cx / W} {cy / H} {8 / W} {6 / H}\n")


class FakeTracker:
    """정답 근처 박스를 내는 가짜 트래커: 30프레임에서 트랙이 끊겨 새 번호(2)로 바뀌고, 오탐 트랙(9)이 있다."""

    def reset(self):
        self.i = 0
        self.rng = np.random.default_rng(1)

    def __call__(self, frame):
        i = self.i
        self.i += 1
        out = []
        cx, cy = obj_pos(i)
        if i < 30:
            out.append([1, cx + self.rng.normal(0, 0.5), cy, 8, 6, 0.8, 14])
        elif i > 33:
            out.append([2, cx + self.rng.normal(0, 0.5), cy, 8.5, 6, 0.7, 14])
        if 10 <= i < 25:
            out.append([9, 130, 100, 5, 5, 0.05, 14])  # 제자리 오탐
        return np.array(out) if out else np.zeros((0, 7))


@pytest.fixture
def data(tmp_path):
    make_clip(tmp_path, "V_BIRD_001", cls=0)
    (tmp_path / "clips.txt").write_text("V_BIRD_001\n")
    return tmp_path


def common_args(d, mode, out):
    return [mode, "--clips", str(d / "clips.txt"), "--source", "vis", "--out", str(d / out),
            "--frames-dir", str(d / "frames"), "--labels-dir", str(d / "labels"), "--fps", "30"]


def test_gt_mode(data):
    extract_tracks.main(common_args(data, "gt", "gt"))
    t = pd.read_parquet(data / "gt" / "tracks.parquet")
    assert t["track_id"].nunique() == 1
    assert len(t) == N and (t["label"] == "bird").all() and t["is_gt"].all()
    assert t["t_sec"].iloc[-1] == pytest.approx((N - 1) / 30)


def test_yolo_mode_labels(data):
    extract_tracks.main(common_args(data, "yolo", "yolo") + ["--weights", "x"], tracker=FakeTracker())
    s = pd.read_csv(data / "yolo" / "track_summary.csv").set_index("track_id")
    assert s.loc["V_BIRD_001:1", "label"] == "bird"
    assert s.loc["V_BIRD_001:2", "label"] == "bird"
    assert s.loc["V_BIRD_001:9", "label"] == "none"
    g = pd.read_csv(data / "yolo" / "gt_objects.csv")
    assert g["n_tracks"].iloc[0] == 2 and g["n_matched"].iloc[0] == N - 4


def test_yolo_mode_stitch(data):
    extract_tracks.main(common_args(data, "yolo", "st") + ["--weights", "x", "--stitch"], tracker=FakeTracker())
    s = pd.read_csv(data / "st" / "track_summary.csv")
    assert sorted(s["label"]) == ["bird", "none"]
    assert pd.read_csv(data / "st" / "gt_objects.csv")["n_tracks"].iloc[0] == 1


def test_eval_guard_blocks_pair(data):
    (data / "eval.txt").write_text("IR_BIRD_001\n")  # 적외선 쌍이 평가면 가시광도 막는다
    with pytest.raises(SystemExit) as e:
        extract_tracks.main(common_args(data, "gt", "x") + ["--eval-ids", str(data / "eval.txt")])
    assert e.value.code == 1


def test_noise_roundtrip(data):
    extract_tracks.main(common_args(data, "yolo", "yolo") + ["--weights", "x"], tracker=FakeTracker())
    extract_tracks.main(common_args(data, "gt", "gt"))
    noise.main(["calibrate", "--tracks", str(data / "yolo" / "tracks.parquet"),
                "--gt-objects", str(data / "yolo" / "gt_objects.csv"), "--out", str(data / "noise.json")])
    noise.main(["apply", "--tracks", str(data / "gt" / "tracks.parquet"), "--noise", str(data / "noise.json"),
                "--out", str(data / "noisy"), "--draws", "2"])
    t = pd.read_parquet(data / "noisy" / "tracks.parquet")
    assert set(t["draw"]) == {0, 1} and not t["is_gt"].any() and (t["label"] == "bird").all()
    assert len(t) < 2 * N  # 누락·끊김으로 일부 프레임이 빠진다


def test_link_gt_two_objects():
    rows = []
    for f in range(20):
        rows += [(f, 0, 10 + f, 10, 4, 4), (f, 1, 100 - f, 50, 6, 6)]
    gt = link_gt(pd.DataFrame(rows, columns=["frame", "cls", "cx", "cy", "w", "h"]))
    assert gt.groupby("gt_obj")["cls"].nunique().max() == 1 and gt["gt_obj"].nunique() == 2


def test_stitch_ignores_far_track():
    a = pd.DataFrame({"track_id": "a", "frame": range(10), "cx": np.arange(10.0), "cy": 0.0,
                      "w": 4.0, "h": 4.0, "conf": 0.9})
    far = a.assign(track_id="b", frame=a["frame"] + 12, cx=500.0)
    near = a.assign(track_id="c", frame=a["frame"] + 12, cx=a["cx"] + 12)
    out = stitch_tracks(pd.concat([a, far, near]), max_gap=5)
    assert out.groupby("track_id").size().to_dict() == {"a": 20, "b": 10}


def test_background_motion_sign():
    rng = np.random.default_rng(0)
    img = (rng.random((128, 160)) * 255).astype(np.uint8)
    bgm = BackgroundMotion(scale=1.0)
    bgm(img)
    dx, dy, _ = bgm(np.roll(img, 4, axis=1))  # 화면 내용이 오른쪽으로 4px
    assert dx == pytest.approx(4, abs=0.5) and dy == pytest.approx(0, abs=0.5)
