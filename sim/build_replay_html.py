"""
build_replay_html.py - 리플레이 JSON(record_replay.py 출력)에 표시 이름과 요약을 붙여 뷰어 HTML로 만든다

  python sim/build_replay_html.py replay.json out.html [--labels labels.json] [--story story.json]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def write_html(result, path):
    with open(os.path.join(HERE, "replay_viewer_template.html"), encoding="utf-8") as f:
        page = f.read()
    data = json.dumps(result, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page.replace("/*__DATA__*/null", data))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("out")
    ap.add_argument("--labels", default="", help="JSON 파일 {policy: [짧은 이름, 긴 이름]}")
    ap.add_argument("--story", default="", help="JSON 파일 {title, sub, table_title, insights: [{title, big, text}], note}")
    args = ap.parse_args()
    with open(args.replay, encoding="utf-8") as f:
        result = json.load(f)
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            labels = json.load(f)
        result["policy_meta"] = [{"key": k, "label": labels.get(k, [k, k])[0], "long": labels.get(k, [k, k])[1]}
                                 for k in result["policies"]]
    if args.story:
        with open(args.story, encoding="utf-8") as f:
            result["story"] = json.load(f)
    write_html(result, args.out)
    print(args.out, os.path.getsize(args.out), "bytes")
