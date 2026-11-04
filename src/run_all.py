"""Reproduce every output in order. Run: python -m src.run_all

Later stages (drivers, forecast, backtest) are added here as they are built.
"""
import json

from src import drivers, prepare


def main() -> None:
    print("Stage 1 - prepare tidy tables")
    report = prepare.main()
    print(json.dumps({k: report[k] for k in ("model_window_months", "model_window_revenue",
                                            "repeat_rate_model_window")}, indent=2))
    print("Stage 2 - driver tree by market")
    print(json.dumps(drivers.main(), indent=2))


if __name__ == "__main__":
    main()
