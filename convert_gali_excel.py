#!/usr/bin/env python3
"""Convert the GALI Excel workbook into per-sheet CSV files."""

import argparse
import csv
from pathlib import Path

import pandas as pd


def sanitize_sheet_name(sheet_name: str) -> str:
    """Return a filesystem-friendly version of the sheet name."""
    return ''.join(char if char.isalnum() else '_' for char in sheet_name.strip()).strip('_').lower()


def convert_workbook(xlsx_path: Path, output_dir: Path, quoting: int) -> None:
    """Export each worksheet in the Excel workbook as a CSV file."""
    xls = pd.ExcelFile(xlsx_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        safe_sheet = sanitize_sheet_name(sheet_name) or 'sheet'
        output_path = output_dir / f"{xlsx_path.stem}_{safe_sheet}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8', quoting=quoting)
        print(f"Saved sheet '{sheet_name}' to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source Excel workbook (.xlsx)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory that will receive the CSV exports (default: ./data)",
    )
    parser.add_argument(
        "--quote",
        choices=["minimal", "all", "nonnumeric", "none"],
        default="minimal",
        help=(
            "CSV quoting level; use 'all' if you expect many embedded commas. "
            "Defaults to the standard minimal quoting."
        ),
    )
    return parser


QUOTE_MAP = {
    "minimal": csv.QUOTE_MINIMAL,
    "all": csv.QUOTE_ALL,
    "nonnumeric": csv.QUOTE_NONNUMERIC,
    "none": csv.QUOTE_NONE,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    convert_workbook(args.input, args.output_dir, QUOTE_MAP[args.quote])


if __name__ == "__main__":
    main()
