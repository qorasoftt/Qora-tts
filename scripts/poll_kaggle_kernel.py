from __future__ import annotations

import argparse
import json
import subprocess
import time

FINAL_STATUSES = {
    "KernelWorkerStatus.COMPLETE",
    "KernelWorkerStatus.ERROR",
    "KernelWorkerStatus.CANCEL",
    "KernelWorkerStatus.FAILED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Kaggle kernel status until completion.")
    parser.add_argument("--kernel", required=True, help="Kernel ref in owner/slug form.")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=30,
        help="Seconds between status checks.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Maximum seconds to wait before failing.",
    )
    return parser.parse_args()


def read_status(kernel: str) -> str:
    completed = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True,
        text=True,
        check=True,
    )
    line = completed.stdout.strip()
    if '"' not in line:
        raise RuntimeError(f"Unexpected Kaggle status output: {line}")
    return line.split('"')[1]


def main() -> int:
    args = parse_args()
    deadline = time.time() + args.timeout_seconds
    history: list[str] = []

    while True:
        status = read_status(args.kernel)
        history.append(status)
        print(json.dumps({"kernel": args.kernel, "status": status}))
        if status in FINAL_STATUSES:
            if status == "KernelWorkerStatus.COMPLETE":
                return 0
            raise SystemExit(f"Kaggle kernel finished with non-success status: {status}")
        if time.time() >= deadline:
            raise SystemExit(
                f"Timed out waiting for {args.kernel}. Last status: {status}. History: {history}"
            )
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
