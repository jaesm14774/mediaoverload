"""Isolated, persistent WanGP Python-API host. Only JSON events use stdout."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="4")
    parser.add_argument("--attention", default="sage2")
    args = parser.parse_args()
    protocol = sys.stdout

    def emit(payload: dict) -> None:
        protocol.write("WAN2GP_EVENT " + json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        protocol.flush()

    # Libraries print during imports and generation. Keep the protocol separate.
    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, str(args.root))
        started = time.perf_counter()
        from shared.api import init
        session = init(root=args.root, output_dir=args.output, cli_args=["--profile", args.profile, "--attention", args.attention, "--verbose", "2"], console_isatty=False)
        emit({"kind": "ready", "startup_seconds": time.perf_counter() - started})
        try:
            for line in sys.stdin:
                request = json.loads(line)
                if request.get("action") == "close":
                    break
                started = time.perf_counter()
                try:
                    if request.get("action") == "schema":
                        model = request["model_type"]
                        emit({"kind": "result", "success": True, "schema": session.get_model_schema(model), "defaults": session.get_default_settings(model)})
                        continue
                    job = session.submit_task(request["settings"])
                    for event in job.events.iter(timeout=0.2):
                        if event.kind == "progress":
                            progress = {key: getattr(event.data, key, None) for key in ("phase", "progress", "current_step", "total_steps", "status")}
                            emit({"kind": "progress", "elapsed_seconds": time.perf_counter() - started, **progress})
                    result = job.result()
                    emit({"kind": "result", "success": result.success, "wall_seconds": time.perf_counter() - started, "generated_files": [str(path) for path in result.generated_files], "errors": [error.message for error in result.errors]})
                except Exception as exc:
                    traceback.print_exc()
                    emit({"kind": "result", "success": False, "wall_seconds": time.perf_counter() - started, "errors": [str(exc)]})
                finally:
                    # Public close unloads models, while retaining the initialized runtime.
                    # Krea and local vision review must be able to use the same 8GB GPU.
                    session.close()
        finally:
            session.close()


if __name__ == "__main__":
    main()
