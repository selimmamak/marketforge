"""Command-line workflows: fetch, benchmark, inspect, and forecast."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig, load_config
from .data import download_yahoo, generate_synthetic, load_csv
from .experiment import forecast_latest, run_experiment, write_json


def _parser():
    parser = argparse.ArgumentParser(
        prog="marketforge", description="Reproducible stock forecasting experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download adjusted daily OHLCV from Yahoo Finance")
    fetch.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "SPY", "QQQ"])
    fetch.add_argument("--start", default="2016-01-01")
    fetch.add_argument("--end", required=True, help="Exclusive end date YYYY-MM-DD")
    fetch.add_argument("--output", type=Path, default=Path("data/market.csv"))
    run = sub.add_parser("run", help="Evaluate a CSV using nested time-ordered validation")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--models", help="Comma-separated registry names (zero is always included)")
    run.add_argument(
        "--resume", action="store_true", help="Resume verified checkpoints in the output directory"
    )
    run.add_argument(
        "--workers", type=int, default=1, help="Independent spawned fit workers (default: 1)"
    )
    demo = sub.add_parser("demo", help="Offline synthetic smoke experiment, not market evidence")
    demo.add_argument("--days", type=int, default=900)
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--models", default="zero,historical_mean,ridge,extra_trees,regime_mixture")
    demo.add_argument("--resume", action="store_true", help="Resume verified demo checkpoints")
    demo.add_argument(
        "--workers", type=int, default=1, help="Independent spawned fit workers (default: 1)"
    )
    forecast = sub.add_parser("forecast", help="Predict latest rows from a trusted local artifact")
    forecast.add_argument("--data", type=Path, required=True)
    forecast.add_argument("--artifact", type=Path, required=True, help="Trusted local model.joblib")
    forecast.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "fetch":
            prices = download_yahoo(args.tickers, args.start, args.end)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            prices.to_csv(args.output, index=False)
            metadata = {
                "provider": "Yahoo Finance via yfinance",
                "adjusted": True,
                "tickers": args.tickers,
                "requested_start": args.start,
                "requested_end_exclusive": args.end,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "actual_start": str(prices.date.min()),
                "actual_end": str(prices.date.max()),
                "rows": len(prices),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
            write_json(args.output.with_suffix(".metadata.json"), metadata)
            print(f"Saved {len(prices):,} rows to {args.output}")
        elif args.command in {"run", "demo"}:
            if args.command == "demo":
                config = ExperimentConfig(
                    quick=True,
                    holdout_size=42,
                    calibration_size=42,
                    test_size=42,
                    inner_splits=1,
                    bootstrap_samples=100,
                )
                prices = generate_synthetic(n_days=args.days, seed=config.seed)
                source = "synthetic demonstration; not financial performance evidence"
            else:
                config = load_config(args.config)
                prices = load_csv(args.data)
                source = str(args.data)
            if args.models:
                config.models = [name.strip() for name in args.models.split(",") if name.strip()]
                config.__post_init__()
            summary = run_experiment(
                prices,
                config,
                args.output,
                source,
                resume=args.resume,
                workers=args.workers,
            )
            print(f"Development-selected model: {summary['winner']}")
            print(f"Holdout RMSE: {summary['holdout_metrics']['rmse']:.6f}")
            print(f"Report: {args.output / 'report.html'}")
            if summary["model_errors"]:
                print(f"Excluded model errors: {summary['model_errors']}", file=sys.stderr)
        elif args.command == "forecast":
            result = forecast_latest(load_csv(args.data), args.artifact)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(result.to_string(index=False))
        return 0
    except (ValueError, RuntimeError, ImportError, OSError, FloatingPointError) as exc:
        print(f"marketforge: {exc}", file=sys.stderr)
        return 2
