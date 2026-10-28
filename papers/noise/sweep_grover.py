r'''
    Error-rate sweep: lumping of noisy Grover against the noise strength.

    For a fixed register size, runs the same lumping as `stress_grover.py` at a geometric grid of
    ``epsilon`` (the scaling of the ibm_boston error rates), plus `0` (noise-free) and `1` (the
    device as reported), up to the point where CZ's depolarising parameter is 1. One CSV row per
    ``epsilon``; the file is rewritten on every run.
'''
from __future__ import annotations
import argparse, sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..")) # clue is here

from circuits import epsilon_grid, scaled_error_rates
from results import append_row
from stress_grover import run_size, NOISE_MODEL

CSV_HEADER = [
    "qubits", "epsilon", "lambda_cz", "lambda_sx", "lambda_x", "ambient_dim", "reduced_dim",
    "lumping_factor", "red_ratio", "time_lumping", "memory_mb", "delta", "target", "iterations",
    "gate_count", "prep_gate_count", "noise_model",
]

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qubits", type=int, default=2, help="register size (default: 2)")
    parser.add_argument("--points", type=int, default=25, help="number of geometrically spaced epsilon values; 0 and 1 are added on top (default: 25)")
    parser.add_argument("--eps-min", type=float, default=1e-5, help="smallest non-zero epsilon (default: 1e-5); the largest is where CZ's depolarising parameter reaches 1")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--csv", type=str, default=None, help="output CSV, overwritten (default: results/sweep-grover-q<qubits>.csv next to this script)")
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(SCRIPT_DIR, "results", f"sweep-grover-q{args.qubits}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"Error-rate sweep: noisy Grover, {args.qubits} qubits, delta={args.delta:g} -> {csv_path}")
    print(f"{'epsilon':>12} {'lambda_cz':>12} {'lumped':>8} {'ambient':>8} {'factor':>9} {'time (s)':>10}")
    sys.stdout.flush()

    for epsilon in epsilon_grid(args.points, args.eps_min):
        result = run_size(args.qubits, epsilon, args.delta)
        rates = scaled_error_rates(epsilon)
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{epsilon:>12.4g} {rates['cz']:>12.4g} {result['reduced_dim']:>8} "
            f"{result['ambient_dim']:>8} {factor:>8.2f}x {result['time']:>10.4f}"
        )
        sys.stdout.flush()
        append_row(csv_path, CSV_HEADER, [
            args.qubits, epsilon, rates["cz"], rates["sx"], rates["x"], result["ambient_dim"],
            result["reduced_dim"], factor, 1 / factor, result["time"], result["memory"], args.delta,
            result["target"], result["iterations"], result["gate_count"], result["prep_gate_count"],
            NOISE_MODEL,
        ])

if __name__ == "__main__":
    main()
