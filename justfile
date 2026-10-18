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
