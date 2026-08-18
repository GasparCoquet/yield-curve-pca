"""Download the Treasury par yield curve and cache it under data/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ycpca import data


def main() -> None:
    panel = data.download(start_year=2015, end_year=2026)
    complete = panel.dropna(how="any")
    path = data.save(complete)
    print(f"rows downloaded : {len(panel)}")
    print(f"rows complete   : {len(complete)}")
    print(f"range           : {complete.index.min().date()} to {complete.index.max().date()}")
    print(f"tenors          : {list(complete.columns)}")
    print(f"cached to       : {path}")


if __name__ == "__main__":
    main()
