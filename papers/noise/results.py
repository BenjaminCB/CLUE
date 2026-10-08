from __future__ import annotations
import csv, os

def append_row(path: str, header: list[str], row: list) -> None:
    r'''
        Append ``row`` to ``path``, writing ``header`` first if the file is new.

        If the file exists with a *different* header we refuse rather than append. Gate-level runs
        changed what ``epsilon`` means (per gate, not per block) and added columns, so silently
        mixing them into an older results file would corrupt the dataset in a way that is invisible
        afterwards.
    '''
    new_file = not os.path.exists(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not new_file:
        with open(path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing is not None and existing != [str(column) for column in header]:
            raise ValueError(
                f"{path} already has header {existing}, which does not match {header}. "
                "Append this run to a different file."
            )
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)
