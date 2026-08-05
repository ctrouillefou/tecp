#!/usr/bin/env python3
# =====================================================================
# EXTRACT_GAMMA_CRIT — AUTOMATIC EXTRACTION OF γ_CRIT FROM 1D SWEEPS
# 
# Description:
#   Post‑processing script that reads all sweep_gamma_1D_speed*_summary.csv
#   files and identifies the bifurcation threshold γ_crit using the
#   20 % threshold rule (the threshold is defined as 20 % above the
#   plateau value at γ = 0.03). The results are saved to a CSV file
#   for further use (e.g., for generating Figure 10).
#
# Version: 1.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. Scans all summary CSV files in the current directory.
#   2. Extracts γ_crit using the 20 % threshold rule.
#   3. Saves results to gamma_crit_summary.csv.
# =====================================================================

import numpy as np
import pandas as pd
import glob
import sys

# ---------------------------------------------------------------------
# 1. Core function: extract gamma_crit using the 20 % rule
# ---------------------------------------------------------------------
def extract_gamma_crit_20pct(df):
    """
    Extract gamma_crit from a 1D sweep summary DataFrame using the 20 % rule.
    The plateau is defined as the CO2 value at gamma = 0.03.
    gamma_crit is the gamma where CO2 first exceeds plateau * 1.20.
    If no such gamma exists, return the extrapolated value.
    """
    # Ensure gamma is sorted
    df = df.sort_values('gamma')
    gamma = df['gamma'].values
    co2 = df['CO2_final_30d'].values

    # Plateau at gamma = 0.03
    if 0.03 not in gamma:
        print("   Warning: gamma=0.03 not found. Using the smallest gamma as plateau.")
        idx_plateau = 0
    else:
        idx_plateau = np.where(gamma == 0.03)[0][0]

    plateau = co2[idx_plateau]
    threshold = plateau * 1.20

    # Find the first gamma where CO2 exceeds the threshold
    for i in range(idx_plateau, len(gamma)):
        if co2[i] > threshold:
            # Interpolate between the previous point and this one
            if i > idx_plateau:
                gamma1 = gamma[i-1]
                gamma2 = gamma[i]
                co2_1 = co2[i-1]
                co2_2 = co2[i]
                if co2_2 - co2_1 > 0:
                    gamma_crit = gamma1 + (gamma2 - gamma1) * (threshold - co2_1) / (co2_2 - co2_1)
                else:
                    gamma_crit = gamma1
            else:
                gamma_crit = gamma[i]
            return gamma_crit, plateau, threshold

    # If no threshold crossing, extrapolate linearly using the last two points
    if len(gamma) >= 2:
        gamma1 = gamma[-2]
        gamma2 = gamma[-1]
        co2_1 = co2[-2]
        co2_2 = co2[-1]
        slope = (co2_2 - co2_1) / (gamma2 - gamma1)
        if slope > 0:
            gamma_crit = gamma2 + (threshold - co2_2) / slope
        else:
            gamma_crit = gamma[-1]  # fallback
        return gamma_crit, plateau, threshold

    # Fallback
    return None, plateau, threshold

# ---------------------------------------------------------------------
# 2. Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("EXTRACT GAMMA_CRIT FOR MULTIPLE SIGNAL SPEEDS (20 % rule)")
    print("=" * 60)

    # Find all summary CSV files from 1D sweeps
    file_pattern = "sweep_gamma_1D_speed*_summary.csv"
    files = glob.glob(file_pattern)

    if not files:
        print(f"❌ No files found matching '{file_pattern}'.")
        print("   Please run 1D sweeps first (e.g., sweep_gamma_1D_V10.py 64).")
        sys.exit(1)

    print(f"📁 Found {len(files)} summary file(s):")
    for f in files:
        print(f"   - {f}")

    results = []  # list of dicts: speed, gamma_crit, plateau, threshold

    for filepath in sorted(files):
        # Extract speed from filename
        speed_str = filepath.split('speed')[1].split('_')[0]
        try:
            speed = float(speed_str)
        except ValueError:
            speed = speed_str

        print(f"\n🔍 Processing speed = {speed}")
        df = pd.read_csv(filepath)

        # Ensure required columns exist
        if 'gamma' not in df.columns or 'CO2_final_30d' not in df.columns:
            print(f"   ❌ File {filepath} missing required columns ('gamma', 'CO2_final_30d').")
            continue

        gamma_crit, plateau, threshold = extract_gamma_crit_20pct(df)

        if gamma_crit is not None:
            results.append({
                'speed': speed,
                'gamma_crit': gamma_crit,
                'plateau': plateau,
                'threshold': threshold
            })
            print(f"   ✅ gamma_crit = {gamma_crit:.6f}  (plateau = {plateau:.6f}, threshold = {threshold:.6f})")
        else:
            print(f"   ❌ No gamma_crit extracted (fallback).")

    # -----------------------------------------------------------------
    # 3. Save results
    # -----------------------------------------------------------------
    if results:
        df_results = pd.DataFrame(results)
        df_results.sort_values('speed', inplace=True)

        # Save summary
        df_results.to_csv('gamma_crit_summary.csv', index=False)
        print("\n📁 Summary saved: gamma_crit_summary.csv")
        print(df_results.to_string(index=False))

    else:
        print("\n❌ No gamma_crit values extracted from any file.")

    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)