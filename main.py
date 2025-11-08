"""
nc2csv_torax.py

Convert TORAX NetCDF (.nc) output to CSV.

Examples:
  # Convert a specific file's profiles group to CSV
  python nc2csv_torax.py /tmp/torax_results/state_history_20251107_220619.nc

  # Auto-pick the latest TORAX file and export profiles + scalars
  python nc2csv_torax.py --latest --write-scalars

  # Export only a few variables (if present) to keep the CSV small
  python nc2csv_torax.py --latest -o torax_profiles_small.csv --vars temperature density current
"""

import argparse
import glob
import os
import sys


def pick_engine():
    # Prefer h5netcdf on macOS; fall back to netcdf4
    try:
        import h5netcdf  # noqa: F401

        return "h5netcdf"
    except Exception:
        return "netcdf4"


def latest_nc(pattern="/tmp/torax_results/state_history_*.nc"):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .nc files found with pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def list_groups(path):
    from netCDF4 import Dataset

    root = Dataset(path, "r")
    groups = list(root.groups.keys())
    root.close()
    return groups


def open_group_with_data(path, preferred_group="profiles"):
    import xarray as xr

    engine = pick_engine()

    # Try preferred group first
    try:
        ds = xr.open_dataset(path, engine=engine, group=preferred_group)
        if len(ds.data_vars) > 0:
            return preferred_group, ds
    except Exception:
        pass

    # Otherwise scan for the first non-empty group
    try:
        from netCDF4 import Dataset

        root = Dataset(path, "r")
        for g in root.groups.keys():
            try:
                ds_try = xr.open_dataset(path, engine=engine, group=g)
                if len(ds_try.data_vars) > 0:
                    root.close()
                    return g, ds_try
            except Exception:
                continue
        root.close()
    except Exception as e:
        raise RuntimeError(f"Could not inspect groups: {e}")

    raise RuntimeError(
        "No non-empty groups with data variables were found in this file."
    )


def to_csv(ds, out_csv, only_vars=None):
    if only_vars:
        keep = [v for v in only_vars if v in ds]
        if not keep:
            raise ValueError("None of the requested --vars are present in the dataset.")
        ds = ds[keep]
    df = ds.to_dataframe().reset_index()
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  (rows={len(df):,}, cols={len(df.columns)})")


def main():
    ap = argparse.ArgumentParser(
        description="Convert TORAX .nc outputs to CSV (profiles + optional scalars)."
    )
    ap.add_argument("path", nargs="?", help="Path to a .nc file. Omit with --latest.")
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest /tmp/torax_results/state_history_*.nc",
        default=True,
    )
    ap.add_argument(
        "-o",
        "--out",
        default="torax_profiles.csv",
        help="Output CSV for profiles (default: torax_profiles.csv)",
    )
    ap.add_argument(
        "--vars",
        nargs="*",
        help="Subset of variables to export (e.g., temperature density current)",
    )
    ap.add_argument(
        "--write-scalars",
        action="store_true",
        help="Also export scalars group to torax_scalars.csv",
    )
    ap.add_argument("--list-groups", action="store_true", help="List groups and exit")
    args = ap.parse_args()

    # Resolve file path
    if args.latest:
        path = latest_nc()
    elif args.path:
        path = args.path
    else:
        ap.error("Provide a path to a .nc file or use --latest.")

    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    if args.list_groups:
        print("Groups:", list_groups(path))
        return

    # Open profiles (or first non-empty) and write CSV
    group_name, ds = open_group_with_data(path, preferred_group="profiles")
    print(f"Using group: {group_name}")
    if args.vars:
        print(f"Selecting variables: {args.vars}")
    to_csv(ds, args.out, only_vars=args.vars)


if __name__ == "__main__":
    main()
