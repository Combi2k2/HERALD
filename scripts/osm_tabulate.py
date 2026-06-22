#!/usr/bin/env python3
"""Export raw osm.geojson features to a tag table (CSV, no geometry)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from herald.data import RunPaths
from herald.data.tabulate import load_osm_geojson, osm_features_to_rows, write_osm_table


def _latest_run_id(data_dir: Path) -> str:
    runs = sorted(
        (p for p in data_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No runs under {data_dir}")
    return runs[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        help="Run folder under data/ (default: newest run)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to osm.geojson (overrides --run-id)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: same dir as input, osm.csv)",
    )
    args = parser.parse_args()

    if args.input is not None:
        osm_path = args.input
        out_path = args.output or osm_path.with_suffix(".csv")
    else:
        run_id = args.run_id or _latest_run_id(Path("data"))
        paths = RunPaths(run_id=run_id)
        osm_path = paths.osm
        out_path = args.output or (paths.raw / "osm.csv")

    if not osm_path.is_file():
        print(f"Error: not found: {osm_path}", file=sys.stderr)
        raise SystemExit(1)

    geojson = load_osm_geojson(osm_path)
    rows = osm_features_to_rows(geojson)
    write_osm_table(rows, out_path)

    print(f"Features: {len(rows)}")
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
