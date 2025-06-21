#!/usr/bin/env python
"""Utility script to convert a `final_result.json` file (from deed-ocr workflow)
into a corresponding Excel workbook.

Usage
-----
python scripts/convert_final_result_to_excel.py <final_result.json | directory> [-o output.xlsx] [--instruction instruction.csv]

If a directory is supplied, the script will recursively search for every
`final_result.json` file inside that directory and convert each one to an Excel
workbook (saved alongside the JSON file). The `-o/--output` flag is only
respected when a single JSON file is processed.

If *output.xlsx* is not supplied, an `.xlsx` file with the same base name will
be created alongside the JSON file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure project root is on sys.path so `deed_ocr` package can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deed_ocr.utils.excel_converter import convert_final_result_to_excel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert final_result.json to Excel. Accepts either a single JSON file "
            "or a directory that contains many sub-folders with final_result.json files."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a final_result.json file or a directory containing multiple such JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination .xlsx file (defaults to <input_path>.xlsx).",
    )
    parser.add_argument(
        "--instruction",
        type=Path,
        default=(Path(__file__).resolve().parent.parent / "instruction.csv"),
        help="Path to instruction.csv (defaults to repo root instruction.csv).",
    )
    return parser.parse_args(argv)


def _process_single_json(json_path: Path, output_path: Path, instruction_csv: Path | None) -> None:
    """Convert *json_path* into Excel written to *output_path*."""
    # Load the JSON content
    with json_path.open("r", encoding="utf-8") as f:
        final_result = json.load(f)

    convert_final_result_to_excel(final_result, output_path, instruction_csv_path=instruction_csv)

    print(f"Excel saved to: {output_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input_path.exists():
        sys.exit(f"Error: Path not found: {args.input_path}")

    instruction_csv = args.instruction if args.instruction.exists() else None

    # ------------------------------------------------------------------
    # Single file mode
    # ------------------------------------------------------------------
    if args.input_path.is_file():
        if args.input_path.name != "final_result.json":
            print(
                "Warning: input file name is not 'final_result.json'. Proceeding anyway.",
                file=sys.stderr,
            )

        output_path = args.output or args.input_path.with_suffix(".xlsx")
        _process_single_json(args.input_path, output_path, instruction_csv)
        return

    # ------------------------------------------------------------------
    # Directory mode (recursive)
    # ------------------------------------------------------------------
    json_files = list(args.input_path.rglob("final_result.json"))
    if not json_files:
        sys.exit("Error: No final_result.json files found in the provided directory.")

    if args.output is not None:
        print(
            "Warning: --output flag is ignored when processing multiple JSON files.",
            file=sys.stderr,
        )

    for json_path in json_files:
        output_path = json_path.with_suffix(".xlsx")
        _process_single_json(json_path, output_path, instruction_csv)


if __name__ == "__main__":
    main() 