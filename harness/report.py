"""Summary JSON + HTML reports: batch summary and per-episode full-pipeline pages."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from .metrics import aggregate, totals
from .runner import EpisodeResult

_CSS = """
body{font-family:-apple-system,Segoe UI,sans-serif;margin:1.5rem;background:#111;color:#ddd}
h1{font-size:1.3rem} h2{font-size:1.05rem;margin-top:1.5rem;color:#9cf}
table{border-collapse:collapse;font-size:.9rem} td,th{padding:.35rem .8rem;border:1px solid #333;text-align:left}
th{background:#1c1c1c} .num{text-align:right} .ok{color:#7c7} .bad{color:#c77}
small,.meta{color:#888} a{color:#9cf}
details{border:1px solid #333;border-radius:6px;margin:.4rem 0;padding:.4rem .8rem;background:#161616}
summary{cursor:pointer;font-family:ui-monospace,monospace;font-size:.85rem}
summary b{color:#9cf}
.step{display:flex;gap:1rem;flex-wrap:wrap;margin-top:.6rem}
figure{margin:0;text-align:center} figure img{width:192px;height:192px;image-rendering:auto;border:1px solid #333;border-radius:4px}
figcaption{font-size:.72rem;color:#888;margin-top:.2rem}
pre{background:#0d0d0d;border:1px solid #2a2a2a;border-radius:6px;padding:.5rem .7rem;margin:.3rem 0;
    font-family:ui-monospace,monospace;font-size:.78rem;white-space:pre-wrap;word-break:break-word;color:#ccc}
.reason{color:#a8d8a8} .label{font-size:.72rem;color:#888;margin-top:.5rem}
"""

_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>harness report</title><style>{css}</style></head><body>
<h1>harness report — {meta}</h1>
<p><small>{generated}</small></p>
<h2>Totals</h2><table>{totals_rows}</table>
<h2>Per cell</h2><table>{cell_rows}</table>
<h2>Episodes (full pipeline, per step)</h2>
<p class="meta">每集一个页面：逐步展示模型真实输入图像、本体状态、思考、输出动作与奖励。</p>
<table>{episode_rows}</table>
</body></html>"""

_EP_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>episode {name}</title><style>{css}</style></head><body>
<h1>{name}</h1>
<p class="meta">{meta_line}</p>
{steps_html}
</body></html>"""


def _kv_table(d: dict) -> str:
    return "<tr><th>metric</th><th>value</th></tr>" + "".join(
        f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in d.items())


def _cell_table(cells: dict) -> str:
    cols = "cell episodes succ rate steps calls tok_in tok_out errors".split()
    rows = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    for key, m in cells.items():
        rate = f"<span class='{'ok' if m['success_rate'] > 0 else 'bad'}'>{m['success_rate']}</span>"
        vals = [key, m["episodes"], m["successes"], rate, m["mean_steps"],
                m["brain_calls"], m["tokens_in"], m["tokens_out"], m["errors"]]
        rows += "<tr>" + "".join(
            f"<td{' class=num' if i and i != 3 else ''}>{v}</td>" for i, v in enumerate(vals)) + "</tr>"
    return rows


def _b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def _figure(path: Path, caption: str) -> str:
    b64 = _b64(path)
    if b64 is None:
        return ""
    return (f"<figure><img src='data:image/png;base64,{b64}' "
            f"loading='lazy'><figcaption>{caption}</figcaption></figure>")


def _step_html(s: dict, frames_dir: Path | None) -> str:
    i = s["step"]
    imgs = ""
    if frames_dir is not None:
        base = _figure(frames_dir / "obs_overlay" / f"{i:04d}_base.png",
                       "模型输入 1 · 第三人称 (叠加CV分割)") or \
            _figure(frames_dir / "obs_base" / f"{i:04d}.png", "模型输入 1 · 第三人称 (128x128)") or \
            _figure(frames_dir / f"f{i:04d}.png", "渲染帧(512, 近似当时第三人称视角)")
        wrist = _figure(frames_dir / "obs_overlay" / f"{i:04d}_wrist.png",
                        "模型输入 2 · 腕部相机 (叠加CV分割)") or \
            _figure(frames_dir / "obs_wrist" / f"{i:04d}.png", "模型输入 2 · 腕部相机")
        imgs = f"<div class='step'>{base}{wrist}</div>"
    state = f"<div class='label'>本体状态文本（输入）</div><pre>{s.get('state_text', '(未记录)')}</pre>"
    perc = s.get("perception") or {}
    if perc:
        plines = []
        for name, m in perc.items():
            if m.get("pos"):
                plines.append(f"{name}: pos=({m['pos'][0]}, {m['pos'][1]}, {m['pos'][2]}) m "
                              f"via {m['camera']} ({m['mask_pixels']} px)")
            elif m.get("last_seen_pos"):
                p = m["last_seen_pos"]
                plines.append(f"{name}: not visible now; last seen "
                              f"{m['last_seen_age_steps']} step(s) ago at "
                              f"({p[0]}, {p[1]}, {p[2]}) m")
            else:
                plines.append(f"{name}: not visible in either camera")
        state += f"<div class='label'>3D 感知 (CV, 输入的一部分)</div><pre>{chr(10).join(plines)}</pre>"
    reason = ""
    if s.get("reasoning"):
        reason = f"<div class='label'>模型思考（reasoning）</div><pre class='reason'>{s['reasoning']}</pre>"
    fails = ""
    if s.get("parse_failures"):
        parts = [f"[{f['stage']}] {f['error']} | 模型原话: {f['raw']}"
                 for f in s["parse_failures"]]
        fails = f"<div class='label'>解析失败轮次（修复根因）</div><pre class='bad'>{chr(10).join(parts)}</pre>"
    raw = f"<div class='label'>模型最终输出</div><pre>{s['raw']}</pre>"
    timing = (f" · brain {s['t_brain_s']}s · env {s['t_env_s']}s"
              if "t_brain_s" in s else "")
    return (f"<details><summary>Step {i:3d} · <b>{s['action']}</b> · "
            f"reward {s['reward']:+.3f} · {'成功' if s['success'] else '未成功'}{timing}</summary>"
            f"{imgs}{state}{fails}{reason}{raw}</details>")


def write_episode_report(ep_json: Path, frames_dir: Path | None, out_path: Path) -> Path:
    d = json.loads(ep_json.read_text())
    r, steps = d["result"], d["steps"]
    meta_line = (f"{r['embodiment']} × {r['task']} · seed {r['seed']} · 状态 {r['status']} · "
                 f"{r['steps']} 步 · {r['brain_calls']} 次调用 · "
                 f"tokens {r['tokens_in']}+{r['tokens_out']} · {r['wall_s']}s"
                 + (f" · {r['error']}" if r.get("error") else ""))
    steps_html = "".join(_step_html(s, frames_dir) for s in steps)
    out_path.write_text(_EP_HTML.format(css=_CSS, name=ep_json.stem, meta_line=meta_line,
                                        steps_html=steps_html))
    return out_path


def write_report(results: list[EpisodeResult], out_dir: Path, meta: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"meta": meta, "totals": totals(results), "cells": aggregate(results)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    ep_rows = ""
    for r in results:
        name = f"{r.embodiment}_{r.task}_{r.seed}"
        ep_json = out_dir / "episodes" / f"{name}.json"
        frames = out_dir / "frames" / name if (out_dir / "frames" / name).exists() else None
        if ep_json.exists():
            page = write_episode_report(ep_json, frames, out_dir / f"episode_{name}.html")
            status = f"<span class='{'ok' if r.status == 'success' else 'bad'}'>{r.status}</span>"
            ep_rows += (f"<tr><td><a href='{page.name}'>{name}</a></td>"
                        f"<td>{status}</td><td class='num'>{r.steps}</td>"
                        f"<td class='num'>{r.brain_calls}</td>"
                        f"<td class='num'>{r.tokens_in + r.tokens_out}</td></tr>")
    html = _HTML.format(css=_CSS, meta=meta or "batch", generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                        totals_rows=_kv_table(summary["totals"]),
                        cell_rows=_cell_table(summary["cells"]), episode_rows=ep_rows)
    (out_dir / "report.html").write_text(html)
    return out_dir / "report.html"
