import argparse
import json

from .io import resolve_config
from .storage import configure_storage


def main():
    parser = argparse.ArgumentParser(description="Route C: VLM geometry + visual sensitivity")
    parser.add_argument("--data-root", help="Large-file root (default: OODEXPLORE_DATA_ROOT or ~/wdl/data)")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download and freeze unique-image ScienceQA subsets")
    prepare.add_argument("--output", help="Default: <data-root>/oodexplore/datasets/scienceqa")
    prepare.add_argument("--dataset", default="derek-thomas/ScienceQA")
    prepare.add_argument("--revision", default="main")
    prepare.add_argument("--dev-count", type=int, default=100)
    prepare.add_argument("--test-count", type=int, default=500, help="0 selects all eligible unique test images")
    prepare.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("run", help="Generate original response and force gray-image replay")
    run.add_argument("--config", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    analysis = sub.add_parser("analyze", help="Fit development ECDF and evaluate frozen test responses")
    analysis.add_argument("--dev", required=True)
    analysis.add_argument("--test", required=True)
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--bootstrap", type=int, default=1000)
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--cases", type=int, default=5)
    inspect = sub.add_parser("show-config", help="Resolve paths without loading model or dataset")
    inspect.add_argument("--config", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        root = configure_storage(args.pop("data_root"), create=command != "show-config")
    except RuntimeError as exc:
        parser.error(str(exc))
    if command == "prepare":
        from .data import prepare as action
        args["output"] = args["output"] or str(root / "oodexplore/datasets/scienceqa")
        action(**args)
    elif command == "run":
        from .pipeline import collect
        config = resolve_config(args.pop("config"))
        config["data_root"] = str(root)
        args["data_path"] = args.pop("data")
        collect(config, **args)
    elif command == "analyze":
        from .analysis import analyze
        args["dev_path"], args["test_path"] = args.pop("dev"), args.pop("test")
        analyze(**args)
    else:
        print(json.dumps({**resolve_config(args["config"]), "data_root": str(root)}, indent=2, ensure_ascii=False))
