#!/usr/bin/env python3
import os
import re
import sys
import argparse
import zipfile
import tempfile
from glob import glob

import pandas as pd

ACTIVITY_PATTERNS = ["walking", "standing", "jumping", "still"]

def infer_activity_from_name(name: str) -> str:
    low = name.lower()
    for a in ACTIVITY_PATTERNS:
        if a in low:
            return a
    return "unknown"

def find_csv_case_insensitive(folder: str, candidates):
    """Return the first path that exists in 'folder' matching any of candidates (case-insensitive)."""
    all_files = {f.lower(): f for f in os.listdir(folder)}
    for cand in candidates:
        for k,v in all_files.items():
            if k == cand.lower():
                return os.path.join(folder, v)
    # fallback: substring match
    for cand in candidates:
        for k,v in all_files.items():
            if cand.lower() in k:
                return os.path.join(folder, v)
    return None

def load_sensor_csv(path: str, rename_map):
    """Load a CSV and rename columns if needed. Expect cols: time, x,y,z + optionally seconds_elapsed."""
    df = pd.read_csv(path)
    # normalize lowercase
    cols_norm = {c.lower().strip(): c for c in df.columns}
    # map expected
    def get_col(*names):
        for n in names:
            if n in cols_norm:
                return cols_norm[n]
        return None

    # Required base columns
    time_col = get_col("time", "timestamp", "ts")
    x_col = get_col("x")
    y_col = get_col("y")
    z_col = get_col("z")

    if time_col is None or x_col is None or y_col is None or z_col is None:
        raise ValueError(f"Missing required columns in {path}. Found: {list(df.columns)}")

    df = df.rename(columns={
        time_col: "timestamp",
        x_col: rename_map["x"],
        y_col: rename_map["y"],
        z_col: rename_map["z"],
    })
    # coerce timestamp to int (nanoseconds)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    # drop rows with NA timestamp
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    return df[["timestamp", rename_map["x"], rename_map["y"], rename_map["z"]]]

def merge_acc_gyr(acc_df: pd.DataFrame, gyr_df: pd.DataFrame, tolerance_ns: int = 2_000_000) -> pd.DataFrame:
    """merge_asof by timestamp with nearest match within tolerance_ns (default 2ms)."""
    acc_df = acc_df.sort_values("timestamp").reset_index(drop=True)
    gyr_df = gyr_df.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        acc_df,
        gyr_df,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance_ns,
    )
    return merged

def process_zip(zip_path: str, output_dir: str, tolerance_ns: int = 2_000_000) -> str:
    base = os.path.basename(zip_path)
    stem = os.path.splitext(base)[0]
    activity = infer_activity_from_name(stem)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # Expected calibrated files
        acc_path = find_csv_case_insensitive(tmpdir, ["Accelerometer.csv"])
        gyr_path = find_csv_case_insensitive(tmpdir, ["Gyroscope.csv"])

        # Fallback to uncalibrated if needed
        if acc_path is None:
            acc_path = find_csv_case_insensitive(tmpdir, ["AccelerometerUncalibrated.csv"])
        if gyr_path is None:
            gyr_path = find_csv_case_insensitive(tmpdir, ["GyroscopeUncalibrated.csv"])

        if acc_path is None or gyr_path is None:
            raise FileNotFoundError(f"Could not find accelerometer/gyroscope CSVs in {zip_path}")

        acc = load_sensor_csv(acc_path, {"x":"acc_x","y":"acc_y","z":"acc_z"})
        gyr = load_sensor_csv(gyr_path, {"x":"gyr_x","y":"gyr_y","z":"gyr_z"})

        merged = merge_acc_gyr(acc, gyr, tolerance_ns=tolerance_ns)
        merged["activity"] = activity

        out_name = f"{stem}_combined.csv"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_name)
        merged.to_csv(out_path, index=False)
        return out_path

def main():
    ap = argparse.ArgumentParser(description="Merge Sensor Logger ZIPs into combined CSVs for HMM pipeline.")
    ap.add_argument("--input_dir", type=str, default=".", help="Folder with .zip files")
    ap.add_argument("--output_dir", type=str, default="./data/test", help="Where to write combined CSVs")
    ap.add_argument("--pattern", type=str, default="*.zip", help="Glob pattern for zip files")
    ap.add_argument("--tolerance_ms", type=float, default=2.0, help="Timestamp matching tolerance in milliseconds")
    args = ap.parse_args()

    tol_ns = int(args.tolerance_ms * 1_000_000)  # convert ms to ns for integer timestamps

    zips = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    if not zips:
        print(f"No ZIP files found in {args.input_dir} with pattern {args.pattern}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    successes, failures = [], []
    for zp in zips:
        try:
            out = process_zip(zp, args.output_dir, tolerance_ns=tol_ns)
            print(f"[OK] {os.path.basename(zp)} -> {out}")
            successes.append((zp, out))
        except Exception as e:
            print(f"[FAIL] {os.path.basename(zp)} :: {e}")
            failures.append((zp, str(e)))

    print("\\nSummary:")
    print(f"  Success: {len(successes)}")
    print(f"  Failures: {len(failures)}")
    if failures:
        for zp, err in failures:
            print(f"   - {os.path.basename(zp)}: {err}")

if __name__ == "__main__":
    main()
