"""Local-only project commands."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="identitygraph")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Folder containing data, artifacts and runtime (default: current folder)")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Generate and evaluate a new immutable synthetic run")
    build.add_argument("--run", default="demo")
    build.add_argument("--seed", type=int, default=41)
    build.add_argument("--accounts", type=int, default=240)
    build.add_argument("--gnn-epochs", type=int, default=12)
    build.add_argument("--skip-gnn", action="store_true", help="Explicit baseline-only smoke run")
    build.add_argument("--core-only", action="store_true", help="Skip extended research experiments explicitly")
    ieee = sub.add_parser("ieee-benchmark", help="Audit or evaluate locally supplied IEEE-CIS CSVs")
    ieee.add_argument("--data-dir", required=True, type=Path)
    ieee.add_argument("--run", default="ieee")
    ieee.add_argument("--audit-only", action="store_true")
    serve = sub.add_parser("serve", help="Start the local investigator application")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.command == "build":
        from .pipeline import build as build_project
        result = build_project(root, args.run, args.seed, args.accounts, args.gnn_epochs, not args.skip_gnn, not args.core_only)
        print(json.dumps(result, indent=2))
    elif args.command == "ieee-benchmark":
        import re
        from .ieee import run_ieee
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,50}", args.run):
            parser.error("Use a simple lowercase run name")
        result = run_ieee(args.data_dir, root / "artifacts" / args.run, args.audit_only)
        print(json.dumps(result, indent=2))
    else:
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
