"""用 python -m nlms_dfe ... 执行，Windows 下保留 main 保护。"""
import argparse
import json

from .common import read_config, project_path, write_json


def main():
    parser = argparse.ArgumentParser(description="NLMS-DFE 神经网络参数选择工具")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "audit"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="configs/default.json")
        if name == "audit":
            sub.add_argument("--output", default="outputs/dataset_audit.json")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path")
    build = commands.add_parser("build-index")
    build.add_argument("directory")
    build.add_argument("--output", required=True)
    split = commands.add_parser("split")
    split.add_argument("source")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--output", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("checkpoint")
    predict.add_argument("channel")
    predict.add_argument("--snr-db", type=float, required=True)
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output")
    args = parser.parse_args()
    if args.command == "train":
        from .training.engine import train
        train(read_config(args.config))
        return
    if args.command in {"inspect", "build-index", "split", "audit"}:
        from .data.tools import inspect_file, build_index, split_index, audit
        if args.command == "inspect":
            result = inspect_file(project_path(args.path))
        elif args.command == "build-index":
            result = build_index(project_path(args.directory), project_path(args.output))
        elif args.command == "split":
            result = split_index(project_path(args.source), project_path(args.output), args.seed)
        else:
            result = audit(read_config(args.config))
            write_json(project_path(args.output), result)
    elif args.command == "evaluate":
        from .training.engine import evaluate
        result = evaluate(project_path(args.checkpoint), project_path(args.output))
    else:
        from .predict import Predictor
        from .data.io import read_field
        predictor = Predictor(project_path(args.checkpoint), args.device)
        data = predictor.checkpoint["config"]["data"]
        h = read_field(project_path(args.channel), data["fields"]["channel"], data["hdf5_matlab_order"])
        result = predictor.predict(h, args.snr_db, args.top_k)
        if args.output:
            write_json(project_path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
