default: run

# Run the noise-paper script in the development environment.
run:
    python papers/noise/script.py

# Run the unit and property-based tests. Extra arguments go to pytest,
# e.g. `just test -k frobenius` or `just test --hypothesis-show-statistics`.
test *args:
    python -m pytest {{ args }}
