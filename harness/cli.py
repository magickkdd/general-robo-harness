"""CLI: `harness list` and `harness run ...`."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .brain import load_api_key, make_brain
from .envs import EMBODIMENTS, SUPPORTED, TASKS, get_adapter, make_cell
from .report import write_report
from .runner import Budget, run_batch


def cmd_list(_args) -> None:
    print(f"{'task':24s} " + " ".join(f"{e:14s}" for e in EMBODIMENTS))
    for t in TASKS:
        row = " ".join("ok            " if (e, t) in SUPPORTED else "-             "
                       for e in EMBODIMENTS)
        print(f"{t:24s} {row}")


def cmd_run(args) -> None:
    if args.all_cells:
        cells = [make_cell(t, e, args.max_steps, gt_state=args.gt) for (e, t) in sorted(SUPPORTED)]
    else:
        cells = [make_cell(args.task, args.embodiment, args.max_steps, gt_state=args.gt)]
    seeds = list(range(args.seed, args.seed + args.episodes))
    budget = Budget(max_episodes=len(cells) * len(seeds),
                    max_brain_calls=args.max_brain_calls,
                    max_tokens=args.max_tokens)

    def make_brain_fn():
        if args.brain == "random":
            return make_brain("random")
        return make_brain("vlm", model=args.model, base_url=args.base_url,
                          api_key=load_api_key(args.api_key_env),
                          max_repair=args.max_repair,
                          thinking=args.thinking == "on",
                          send_images=args.images)

    meta = (f"brain={args.brain}"
            + (f"/{args.model}" if args.brain == "vlm" else "")
            + f" episodes={args.episodes}"
            + (" +gt" if args.gt else "")
            + f" thinking={args.thinking}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = cells[0].embodiment if args.all_cells else cells[0].task
    out_dir = Path(args.out) if args.out else Path("runs") / f"{stamp}_{name}_{args.brain}"
    results = run_batch(lambda: get_adapter("maniskill"), cells, make_brain_fn, seeds,
                        out_dir=out_dir, budget=budget,
                        record_every=1 if args.record_video else args.record_every)
    html = write_report(results, out_dir, meta=meta)
    print(f"report: {html}")


def main() -> None:
    p = argparse.ArgumentParser(prog="harness",
                                description="cross-embodiment, cross-VLM benchmark harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show task x embodiment matrix").set_defaults(func=cmd_list)

    r = sub.add_parser("run", help="run episodes")
    r.add_argument("--task", default="PickCube-v1", choices=list(TASKS))
    r.add_argument("--embodiment", default="panda", choices=EMBODIMENTS)
    r.add_argument("--all-cells", action="store_true", help="run every supported cell")
    r.add_argument("--brain", default="random", choices=["random", "vlm"])
    r.add_argument("--model", default="", help="model name for --brain vlm")
    r.add_argument("--base-url", default="", help="OpenAI-compatible base URL")
    r.add_argument("--api-key-env", default=None, help="env var holding the API key")
    r.add_argument("--episodes", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-brain-calls", type=int, default=2000)
    r.add_argument("--max-tokens", type=int, default=2_000_000)
    r.add_argument("--max-repair", type=int, default=1)
    r.add_argument("--gt", action="store_true",
                   help="privileged ground-truth object state (debug; marked in report)")
    r.add_argument("--images", choices=["all", "base", "none"], default="all",
                   help="camera images sent to the brain")
    r.add_argument("--thinking", choices=["on", "off"], default="off",
                   help="gateway-side model thinking (off = fast, ~3s/step)")
    r.add_argument("--max-steps", type=int, default=None,
                   help="cap episode length (min with the env's own horizon)")
    r.add_argument("--record-every", type=int, default=0, help="save a frame every K steps")
    r.add_argument("--record-video", action="store_true", help="record every step and encode mp4 per episode")
    r.add_argument("--out", default=None, help="output directory (default runs/<stamp>_...)")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    if args.cmd == "run" and args.brain == "vlm" and not (args.model and args.base_url):
        p.error("--brain vlm requires --model and --base-url")
    args.func(args)


if __name__ == "__main__":
    main()
