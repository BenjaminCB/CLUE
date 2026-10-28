default:
    @just --list

# Run the noise-paper script in the development environment.
run:
    python papers/noise/script.py

# Run the unit and property-based tests. Extra arguments go to pytest,
# e.g. `just test -k frobenius` or `just test --hypothesis-show-statistics`.
test *args:
    python -m pytest {{ args }}

# Stress test the noisy quantum bisimulation (lumping) on Grover's algorithm: sweeps the
# register size, printing timing and reduction stats to stdout. Extra arguments go to the
# script, e.g. `just stress-grover --max-qubits 12 --epsilon 1e-2`.
stress-grover *args:
    python -u papers/noise/stress_grover.py {{ args }}

# Stress test the noisy quantum bisimulation (lumping) on the MaxCut cost layer: sweeps
# the graph size, printing timing and reduction stats to stdout. Extra arguments go to the
# script, e.g. `just stress-qaoa-maxcut --density 0.5 --epsilon 0`.
stress-qaoa-maxcut *args:
    python -u papers/noise/stress_qaoa_maxcut.py {{ args }}

# Sweep the error-rate scaling epsilon for noisy Grover at a fixed register size, writing
# lumping factor against error rate to a CSV, e.g. `just sweep-grover --qubits 3 --points 25`.
sweep-grover *args:
    python -u papers/noise/sweep_grover.py {{ args }}

# Sweep the error-rate scaling epsilon for the noisy MaxCut cost layer at a fixed graph size,
# e.g. `just sweep-qaoa-maxcut --qubits 3 --points 25`.
sweep-qaoa-maxcut *args:
    python -u papers/noise/sweep_qaoa_maxcut.py {{ args }}
