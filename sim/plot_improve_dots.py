"""
plot_improve_dots.py - 개선안별 500판 평가 승률 (시드별 점 + 평균)

  python sim/plot_improve_dots.py sim/results/saboteur_improve
"""
import json
import os
import sys

import matplotlib
import numpy as np

from analyze import plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "sim/results/saboteur_improve"
CFG = [("B4", "B: 엔트로피 0.003 (학습률 1e-4)"), ("E1", "E1: B + 관측 v2 + 목격 페널티"),
       ("C4", "C: B + 학습률 3e-4"), ("E7", "E7: C + 행동 막기"), ("E6", "E6: E7 + 관측 v2"),
       ("E4", "E4: C + 관측 v2 + 페널티"), ("E5", "E5: E4 + 행동 막기"), ("E2", "E2: E5 + 묶음 업데이트"),
       ("E3", "E3: E2 + 모방 학습 시작")]
REFS = [(0.422, "규칙봇 사보타주 42%"), (0.174, "방 붙기 17%")]
SURF, INK, INK2, GRID, DOT, RANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#c3c2b7"

r = {}
for f in sorted(os.listdir(os.path.join(OUT, "eval"))):
    if f.startswith("round") and f.endswith(".txt"):
        for line in open(os.path.join(OUT, "eval", f), encoding="utf-8"):
            k, js = line.split(" ", 1)
            r[k] = json.loads(js)["sab_win"]

fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURF)
ax.set_facecolor(SURF)
n = len(CFG)
for i, (k, _) in enumerate(CFG):
    y = n - 1 - i
    v = [r[f"{k}_s0"], r[f"{k}_s1"]]
    ax.plot([min(v), max(v)], [y, y], color=RANGE, lw=2, zorder=1)
    ax.scatter(v, [y, y], s=64, color=DOT, edgecolor=SURF, linewidth=2, zorder=3)
    m = float(np.mean(v))
    ax.plot([m, m], [y - 0.25, y + 0.25], color=INK, lw=2, zorder=4)
    ax.text(max(v) + 0.012, y, f"평균 {m:.0%}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(n))
ax.set_yticklabels([c[1] for c in CFG][::-1], fontsize=9.5, color=INK)
for x, lab in REFS:
    ax.axvline(x, color="#8a8986", lw=1, ls="--", zorder=0)
    ax.text(x + 0.004, -0.85, lab, fontsize=8.5, color=INK2)
ax.axhline(n - 2.5, color=GRID, lw=1)
ax.set_ylim(-1.1, n - 0.5)
ax.set_xlim(0, 0.5)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("사보타주 승률 (강화 규칙봇 인간팀 상대, 4M 스텝, 새 시드 500판)", color=INK2)
ax.set_title("개선안별 사보타주 승률 — 점 = 시드 0·1, 검은 선 = 평균", fontsize=11.5, loc="left", color=INK)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(RANGE)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=INK2)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "win_rates.png"), dpi=120, facecolor=SURF)
print(os.path.join(OUT, "win_rates.png"))
