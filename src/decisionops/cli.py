from .provenance import capture
import argparse
import csv
import json
from pathlib import Path
from .core import simulate, benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report, rows = benchmark(simulate())
    (out / "results.json").write_text(json.dumps(report, indent=2))
    with (out / "decisions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = "# DecisionOps backtest\n\nSynthetic demand, hypothetical shortage/excess costs 9:1, shared daily capacity 1000.\n\n"
    for name, result in report["models"].items():
        r = result["test"]
        text += f"- {name}: MAE {r['mae']:.2f}; mean-capacity cost {r['mean_policy_cost']:.0f}; optimised cost {r['optimised_policy_cost']:.0f}; interval coverage {r['interval_90_coverage']:.3f}.\n"
    text += f"\nSelection uses days 360â€“419 only. Lowest-MAE selection: {report['validation_mae_selected']}; lowest-cost selection: {report['validation_cost_selected']}. No test-based winner selection. Forecast accuracy and decision cost are different objectives; the demo does not guarantee a ranking reversal.\n"
    (out / "summary.md").write_text(text)
    print(text)
    capture(out)


if __name__ == "__main__":
    main()
