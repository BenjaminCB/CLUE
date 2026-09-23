from __future__ import annotations
import csv, os

def append_row(path: str, header: list[str], row: list) -> None:
    new_file = not os.path.exists(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)
