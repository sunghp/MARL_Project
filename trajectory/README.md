# 궤적 데이터 추출

svanstrom 가시광 225클립(val 75 + train 150)과 적외선 225클립(같은 클립의 쌍)에서 궤적 분류기 학습용 트랙을 뽑는다.

| 데이터 | 방식 | 결과 |
|---|---|---|
| 가시광 val 75 | `yolo`: YOLO → 트래커 → 정답 매칭 | 실제 트랙. 라벨 bird / drone / other / **none(오탐)** / ambiguous |
| 가시광 train 150 (선택) | `gt` → `noise apply` | 정답 궤적 + 실측 잡음. YOLO가 학습한 클립이라 YOLO 트랙은 쓰지 않는다 |
| 적외선 225 | `gt` → `noise apply` | 정답 궤적 + 실측 잡음 |

잡음 크기는 가시광 val의 실제 YOLO 트랙에서 잰다(`noise calibrate`).

## 준비

```
pip install -r trajectory/requirements.txt
```

- **클립 목록** (한 줄에 클립 ID 하나): `splits/vis_val.txt`, `splits/vis_train.txt`, `splits/ir_trainval.txt`
- **평가 ID 목록**: `splits/eval_ids.txt`. 입력에 평가 클립이나 그 쌍(V_/IR_ 앞부분만 다른 ID)이 있으면 종료코드 1로 멈춘다.
- **영상**: `--video-dir` 안에 `클립ID.mp4`. 이미지로 풀어 둔 경우 `--frames-dir/클립ID/*.jpg`.
- **라벨**: 팀 클래스 대응표(0 새, 1 드론, 2 기타)로 바꾼 프레임별 YOLO txt.
  - `--label-layout subdir`: `labels/클립ID/000123.txt` (기본)
  - `--label-layout flat`: `labels/클립ID_000123.txt`
  - 파일 이름의 마지막 숫자가 프레임 번호다. 1부터 시작하면 `--frame-offset 1`.
  - YOLO txt 대신 CSV(`clip,frame,cls,x,y,w,h`, 왼쪽 위 기준 px, cls는 bird/drone/airplane/helicopter)도 된다: `--gt-csv`.

## 실행

```bash
cd trajectory

# 1) 가시광 val: 실제 YOLO 트랙 (파인튜닝 전에는 --weights yolo11n.pt 로 개발)
python extract_tracks.py yolo --source vis \
  --clips ../splits/vis_val.txt --eval-ids ../splits/eval_ids.txt \
  --video-dir ../data/svanstrom/Video_V --labels-dir ../data/labels/vis \
  --weights ../runs/detect/train/weights/last.pt \
  --tracker trackers/bytetrack_small.yaml --stitch \
  --out ../out/vis_val

# 2) 적외선 225: 정답 궤적
python extract_tracks.py gt --source ir \
  --clips ../splits/ir_trainval.txt --eval-ids ../splits/eval_ids.txt \
  --video-dir ../data/svanstrom/Video_IR --labels-dir ../data/labels/ir \
  --out ../out/ir_gt

# 3) (선택) 가시광 train 150: 정답 궤적
python extract_tracks.py gt --source vis \
  --clips ../splits/vis_train.txt --eval-ids ../splits/eval_ids.txt \
  --video-dir ../data/svanstrom/Video_V --labels-dir ../data/labels/vis \
  --out ../out/vis_train_gt

# 4) 잡음 재기(1의 결과) → 정답 궤적에 입히기
python noise.py calibrate --tracks ../out/vis_val/tracks.parquet \
  --gt-objects ../out/vis_val/gt_objects.csv --out ../out/noise.json
python noise.py apply --tracks ../out/ir_gt/tracks.parquet --noise ../out/noise.json \
  --draws 3 --out ../out/ir_noisy
python noise.py apply --tracks ../out/vis_train_gt/tracks.parquet --noise ../out/noise.json \
  --draws 3 --out ../out/vis_train_noisy
```

궤적 분류기 학습 데이터 = `out/vis_val` + `out/ir_noisy` (+ `out/vis_train_noisy`). `ambiguous` 트랙은 뺀다.

## 출력

| 파일 | 내용 |
|---|---|
| `tracks.parquet` | 트랙 한 프레임당 한 행 |
| `track_summary.csv` | 트랙당 한 행: 길이, 매칭 비율, 라벨 |
| `gt_objects.csv` | 정답 물체당 한 행: 프레임 수, 매칭된 프레임 수, 걸친 트랙 수 |
| `clip_summary.csv` | 클립당 한 행: 트랙 수, 라벨별 개수, 카메라 확인 지표 |

`tracks.parquet`의 주요 열:

| 열 | 뜻 |
|---|---|
| `source`, `clip_id`, `pair_id` | vis/ir, 클립, 가시광·적외선 쌍 키(교차검증 그룹에 쓴다) |
| `track_id`, `frame`, `t_sec`, `fps` | 트랙 번호(`클립:번호`), 프레임, 초 |
| `cx, cy, w, h`, `img_w, img_h` | 박스 중심·크기(px), 이미지 크기 |
| `conf`, `cls_yolo` | YOLO 점수·클래스(정답 궤적은 1, -1) |
| `gt_obj, gt_cls, gt_cx …` | 그 프레임에 매칭된 정답(없으면 -1/NaN) |
| `bg_dx, bg_dy, bg_resp` | 이전 프레임 대비 배경 이동(px)과 신뢰도. 카메라 이동 보정에 쓴다 |
| `label` | 트랙 라벨: bird / drone / other / none / ambiguous |

## 트랙 라벨 규칙 (yolo 방식)

프레임마다 트랙 박스와 정답 박스를 **중심점 거리**로 1:1 매칭한다(거리 ≤ max(5px, 정답 크기)).

- 트랙 프레임의 50% 이상이 매칭되고, 그중 80% 이상이 같은 정답 물체 → 그 물체의 클래스
- 매칭이 10% 이하 → `none` (오탐 트랙)
- 그 밖 → `ambiguous`

값은 `--pos-ratio`, `--neg-ratio`, `--purity`, `--match-min-px`, `--match-rel`로 바꾼다.

## 돌린 뒤 확인할 것

1. **카메라가 물체를 따라갔나**: `clip_summary.csv`의 `gt_center_ratio`(정답 중심이 화면 가운데 ±15%에 있는 비율)와 `bg_motion_px`(프레임당 배경 이동)가 높으면 `--tracker trackers/botsort_small.yaml`로 다시 돌리고, 궤적 특징은 `bg_dx, bg_dy`로 보정한다.
2. **트랙이 끊기나**: `tracks_per_gt_object`가 1.5를 넘으면 `--stitch`를 켜거나 `--stitch-gap-s`를 늘린다.
3. **none 트랙이 충분한가**: 마지막에 출력되는 라벨별 개수를 본다. 적으면 `--conf`나 트래커 설정의 `new_track_thresh`를 낮춘다.
4. **라벨과 영상 길이가 맞나**: `label_beyond_video`가 True인 클립은 프레임 번호가 어긋난 것이다(`--frame-offset` 확인).

## 테스트

```
python -m pytest trajectory/tests -q
```

합성 영상(움직이는 사각형)과 가짜 트래커로 정답 궤적, 트랙 라벨, 트랙 잇기, 평가 차단, 잡음 보정·적용, 배경 이동을 확인한다.
실제 Ultralytics 트래커 연결(`UltralyticsTracker`)은 테스트에 포함되지 않았다. 첫 실행 때 클립 1~2개로 먼저 확인한다.
