#!/usr/bin/env python3
# =====================================================================
# STROBOSCOPIC_POINCARE_SECTION_FIGURE — REGENERATES FIGURE 6A
# 
# Description:
#   Post‑processing script that reads TECP_V6.2_stroboscopic.csv and
#   regenerates the stroboscopic Poincaré section (Figure 6A) for the
#   manuscript. The trajectories with and without WWW are superimposed,
#   so a single curve is plotted.
#
# Version: 1.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. Reads TECP_V6.2_stroboscopic.csv.
#   2. Plots WL (marsh side) vs WR (continental side).
#   3. Uses a single curve (data with and without WWW are identical).
#   4. Saves output as Figure_V6.2_Poincare_Strobe_REVISED.png (300 dpi).
# =====================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('TECP_V6.2_stroboscopic.csv')

# Convert columns to NumPy arrays (avoids Pandas indexing issues)
WL_off = np.array(df['WL_off'])
WR_off = np.array(df['WR_off'])

# Plot a single curve (data with and without WWW are identical)
plt.figure(figsize=(8, 8))
plt.plot(WL_off, WR_off, 'o-', markersize=4, color='blue', label='With & without WWW (superimposed)')
plt.xlabel('W_L – Marsh side (%)')
plt.ylabel('W_R – Continental side (%)')
plt.title('Stroboscopic Poincaré Section (period = 50 s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_V6.2_Poincare_Strobe_REVISED.png', dpi=300)
plt.close()

print("Figure generated: Figure_V6.2_Poincare_Strobe_REVISED.png")