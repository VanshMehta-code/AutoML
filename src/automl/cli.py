"""Command-line interface: `automl train` and `automl predict`. See docs/01_quickstart.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import Config
from .model import FittedModel
from .trainer import AutoML


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="automl", description="Deployment-oriented AutoML")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Search for a model and save it")
    train.add_argument("--data", required=True, help="Path to a CSV file")
    train.add_argument("--target", help="Target column (overrides the config file)")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--out", default="artifacts", help="Output directory")

    predict = sub.add_parser("predict", help="Score a CSV with a saved model")
    predict.add_argument("--model", required=True, help="Path to model.joblib")
    predict.add_argument("--data", required=True, help="Path to a CSV file (target column optional)")
    predict.add_argument("--out", help="Write probabilities to this CSV (default: stdout)")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        cfg = Config.from_yaml(args.config)
        if args.target:
            cfg.target = args.target
        automl = AutoML(cfg).fit(pd.read_csv(args.data))

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        automl.model_.save(out / "model.joblib")
        (out / "report.json").write_text(json.dumps(automl.model_.report, indent=2, default=str))
        print(f"Saved model and report to {out}/")

    elif args.cmd == "predict":
        model = FittedModel.load(args.model)
        proba = model.predict_proba(pd.read_csv(args.data))
        result = pd.DataFrame(proba, columns=[str(c) for c in model.classes])
        if args.out:
            result.to_csv(args.out, index=False)
        else:
            print(result.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
