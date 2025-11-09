#!/usr/bin/env python3
"""
Batch convert all NetCDF files in a directory to CSV using nc2csv.py
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# Import the conversion functions from nc2csv
sys.path.insert(0, str(Path(__file__).parent))
from nc2csv import open_group_with_data, to_csv


def batch_convert(input_dir, output_dir, only_vars=None, verbose=True):
    """
    Convert all .nc files in input_dir to CSV files in output_dir.

    Args:
        input_dir: Directory containing .nc files
        output_dir: Directory to save CSV files
        only_vars: Optional list of variables to export
        verbose: Print progress
    """
    # Find all .nc files
    pattern = os.path.join(input_dir, "*.nc")
    nc_files = glob.glob(pattern)

    if not nc_files:
        print(f"No .nc files found in {input_dir}")
        return

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(nc_files)} .nc files to convert")
    print(f"Output directory: {output_dir}")
    if only_vars:
        print(f"Exporting only variables: {only_vars}")
    print("-" * 60)

    success_count = 0
    failed_files = []

    for nc_file in sorted(nc_files):
        basename = os.path.basename(nc_file)
        csv_name = basename.replace(".nc", ".csv")
        csv_path = os.path.join(output_dir, csv_name)

        try:
            if verbose:
                print(f"\nProcessing: {basename}")

            # Open and convert
            group_name, ds = open_group_with_data(nc_file, preferred_group="profiles")

            if verbose:
                print(f"  Group: {group_name}")
                print(
                    f"  Variables: {list(ds.data_vars.keys())[:5]}..."
                    if len(ds.data_vars) > 5
                    else f"  Variables: {list(ds.data_vars.keys())}"
                )

            to_csv(ds, csv_path, only_vars=only_vars)
            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed_files.append((basename, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print(f"Conversion Summary")
    print("=" * 60)
    print(f"Total files: {len(nc_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for fname, error in failed_files:
            print(f"  - {fname}: {error}")

    print(f"\nCSV files saved to: {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Batch convert all TORAX .nc files in a directory to CSV"
    )
    ap.add_argument(
        "input_dir",
        nargs="?",
        default="./sim_data",
        help="Directory containing .nc files (default: ./sim_data)",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        default="./sim_data/csv",
        help="Output directory for CSV files (default: ./sim_data/csv)",
    )
    ap.add_argument(
        "--vars",
        nargs="*",
        help="Subset of variables to export (e.g., temperature density current)",
    )
    ap.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )

    args = ap.parse_args()

    if not os.path.exists(args.input_dir):
        sys.exit(f"Input directory not found: {args.input_dir}")

    batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        only_vars=args.vars,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
