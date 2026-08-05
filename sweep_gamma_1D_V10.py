#!/usr/bin/env python3
# =====================================================================
# SWEEP_GAMMA_1D_V10 — ONE‑DIMENSIONAL SWEEP OVER γ AT FIXED SIGNAL SPEED
# 
# Description:
#   Generic script that performs a 1D sweep over γ (0.03 to 0.25) for
#   a given signal speed (passed as command‑line argument). Simulates
#   30 days with 24‑h forcing and extracts the bifurcation curve.
#
# Usage: python3 sweep_gamma_1D_V10.py <speed>
# Example: python3 sweep_gamma_1D_V10.py 64
#
# Version: V10.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. Takes signal speed as argument (4, 8, 16, 32, 64, 128, 168).
#   2. Sweeps γ over [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25].
#   3. Computes final CO₂ and temperature.
#   4. Generates bifurcation and time‑series figures.
#   5. Saves summary and series CSV files.
# =====================================================================

import sys
import json
import csv
import time
import numpy as np
from fipy import (
    CellVariable, Grid3D, TransientTerm, DiffusionTerm, ConvectionTerm
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_masks import build_masks

# ---------------------------------------------------------------------
# 0. ARGUMENT PARSING
# ---------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python3 sweep_gamma_1D_V10.py <speed>")
    print("Example: python3 sweep_gamma_1D_V10.py 64")
    sys.exit(1)

try:
    FIXED_SPEED = float(sys.argv[1])
except ValueError:
    print("❌ Speed must be a number (e.g., 4, 8, 16, 32, 64, 128, 168).")
    sys.exit(1)

# ---------------------------------------------------------------------
# 1. LOGGING
# ---------------------------------------------------------------------
LOG_FILE = f"sweep_gamma_1D_speed{FIXED_SPEED:.0f}.log"
log_handle = open(LOG_FILE, "w", buffering=1)

def log(msg):
    """Write a message to console and log file."""
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("=" * 80)
log(f"SWEEP 1D V10 — GAMMA SWEEP AT FIXED SIGNAL SPEED = {FIXED_SPEED:.0f}")
log("=" * 80)

# ---------------------------------------------------------------------
# 2. FIXED PARAMETERS (30 days, 24h forcing)
# ---------------------------------------------------------------------
BASELINE_SPEED = 168.0         # slowest signal speed tested
TAU_BASE = 3600.0
DT = 360.0                     # 6 min
FORCING_PERIOD = 24.0 * 3600.0 # 24 h
N_PERIODS = 30                 # 30 days
STROBE_STEPS = int(FORCING_PERIOD / DT)
N_STEPS = N_PERIODS * STROBE_STEPS

log(f"Time step: {DT:.1f} s")
log(f"Forcing period: {FORCING_PERIOD/3600:.1f} h")
log(f"Total simulated: {N_STEPS * DT / 86400:.1f} days")

# Physical constants (fixed)
D_gas = 2e-5
M_max = 1.0
D_myco = 1e-4
D_M = 1e-6
chi = 1e-3

# ---------------------------------------------------------------------
# 3. LOAD BASE PARAMETERS (from params_real.json)
# ---------------------------------------------------------------------
try:
    with open('params_real.json', 'r') as f:
        P = json.load(f)
    log("✅ Base parameters loaded from params_real.json")
except FileNotFoundError:
    log("❌ params_real.json not found.")
    sys.exit(1)

# ---------------------------------------------------------------------
# 4. SWEEP GRID
# ---------------------------------------------------------------------
gamma_values = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25]

log(f"Gamma values: {gamma_values}")
log(f"Fixed signal speed: {FIXED_SPEED:.1f}")

# ---------------------------------------------------------------------
# 5. SIMULATION FUNCTION
# ---------------------------------------------------------------------
def run_simulation(gamma_val, speed_factor):
    """
    Run a 30‑day simulation for a given gamma and signal speed factor.
    Returns a dict with CO2 history, final CO2, temperature, etc.
    """
    # Update gamma
    P['gamma'] = gamma_val

    # Adjust signal rates according to speed_factor.
    # Baseline signal speed is 168 (slowest tested). For speed_factor = 64,
    # the signal is accelerated by a factor 168/64 = 2.625.
    scale_ratio = BASELINE_SPEED / speed_factor
    lambda_S_eff = P['lambda_S'] * scale_ratio
    k_prod_eff = P['k_prod'] * scale_ratio
    k_basal_eff = P['k_basal'] * scale_ratio

    # Mesh
    nx = ny = nz = 17
    dx = dy = dz = 0.05
    mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

    # Variables
    theta   = CellVariable(mesh=mesh, value=0.3,  hasOld=True)
    Temp    = CellVariable(mesh=mesh, value=298.0, hasOld=True)
    Cv      = CellVariable(mesh=mesh, value=0.02, hasOld=True)
    O2      = CellVariable(mesh=mesh, value=0.21, hasOld=True)
    CO2     = CellVariable(mesh=mesh, value=0.0004, hasOld=True)
    biomass = CellVariable(mesh=mesh, value=0.1,  hasOld=True)
    S       = CellVariable(mesh=mesh, value=0.01,  hasOld=True)
    M       = CellVariable(mesh=mesh, value=0.1,  hasOld=True)

    # Masks
    geometry_params = {
        'x_split': P['x_split'],
        'z_ceil': P['z_ceil'],
        'z_sol_continental': P['z_sol_continental'],
        'z_plant_max_continental': P['z_plant_max_continental'],
        'z_sol_marsh': P['z_sol_marsh'],
        'z_plant_max_marsh': P['z_plant_max_marsh'],
    }
    M_ceil, M_photo, M_resp, M_transp, M_uptake = build_masks(mesh, geometry_params)
    soil_mask = M_uptake
    ceiling_mask = M_ceil

    # Storage (every 20 steps = every 2 hours)
    record_step = 20
    n_records = N_STEPS // record_step + 1
    hist = {
        'step': np.zeros(n_records, dtype=int),
        'time_h': np.zeros(n_records),
        'CO2': np.zeros(n_records),
        'Temp': np.zeros(n_records),
        'theta': np.zeros(n_records),
        'biomass': np.zeros(n_records),
        'S_mean': np.zeros(n_records),
        'M_mean': np.zeros(n_records),
        'entropy': np.zeros(n_records),
    }

    idx = 0
    for step in range(N_STEPS):
        t = step * DT
        if step % record_step == 0:
            hist['step'][idx] = step
            hist['time_h'][idx] = t / 3600.0
            hist['CO2'][idx] = np.mean(CO2.value)
            hist['Temp'][idx] = np.mean(Temp.value)
            hist['theta'][idx] = np.mean(theta.value)
            hist['biomass'][idx] = np.mean(biomass.value)
            hist['S_mean'][idx] = np.mean(S.value)
            hist['M_mean'][idx] = np.mean(M.value)

            # Entropy production (temperature gradient)
            T_3D = Temp.value.reshape(nx, ny, nz)
            grad_T = np.gradient(T_3D)
            grad_norm2 = sum(g**2 for g in grad_T)
            entropy = np.mean(grad_norm2 / (T_3D**2 + 1e-8))
            hist['entropy'][idx] = entropy
            idx += 1

        # Update old values
        theta.updateOld(); Temp.updateOld(); Cv.updateOld()
        O2.updateOld(); CO2.updateOld(); biomass.updateOld()
        S.updateOld(); M.updateOld()

        # Forcing
        day = 0.5 + 0.5 * np.sin(2 * np.pi * t / FORCING_PERIOD)

        # Condensation
        RH = float(np.mean(Cv.value) / 0.03)
        cond_val = max(0, P['k_cond'] * (RH - 0.6))
        cond_term = CellVariable(mesh=mesh, value=cond_val)

        # Evapotranspiration
        evap = P['evap_coeff'] * (1 + day) * theta * (Temp - 273.0)
        uptake = P['uptake_coeff'] * theta * soil_mask
        transpiration = P['transpiration_coeff'] * biomass * (1 + day) * M_transp

        # Photosynthesis
        f_T = np.maximum(0, np.exp(-((Temp.value - P['T_opt']) / P['T_range'])**2))
        bio_active = np.maximum(0, biomass.value) * soil_mask
        Pmax_eff = P['Pmax'] * (1 + 0.1 * day)
        photosynthesis = (
            Pmax_eff
            * (CO2 / (CO2 + P['k_CO2']))
            * f_T
            * theta
            * bio_active
            * M_photo
        )

        # Respiration
        resp_factor = np.maximum(0.1, 1 + 0.05 * (Temp.value - P['T_opt']))
        respiration = P['R_maint'] * bio_active * resp_factor * M_resp

        # --- WWW (with adjusted signal rates) ---
        stress = np.maximum(0, P['P_seuil'] - photosynthesis.value)
        source_S = (k_prod_eff * stress + k_basal_eff) * soil_mask
        convection_S = ConvectionTerm(coeff=-chi * M.faceGrad, var=S)
        eq_S = (
            TransientTerm(var=S)
            == DiffusionTerm(coeff=D_myco, var=S)
            + convection_S
            + source_S
            - lambda_S_eff * S
        )
        eq_M = (
            TransientTerm(var=M)
            == DiffusionTerm(coeff=D_M, var=M)
            + P['gamma_M'] * S * (1 - M / M_max) * soil_mask
            - P['delta_M'] * M
        )
        S_safe = np.maximum(0, S.value)
        hill_term = S_safe**P['n_hill'] / (P['Km']**P['n_hill'] + S_safe**P['n_hill'] + 1e-8)
        cooperation_term = (P['gamma'] * hill_term * biomass.value * soil_mask) / TAU_BASE

        eq_bio = TransientTerm(var=biomass) == (
            0.6 * photosynthesis
            - respiration
            + cooperation_term
        )

        # Physical PDEs
        eq_theta = TransientTerm() == (
            DiffusionTerm(coeff=1e-6) - uptake - evap + cond_term
        )
        eq_vapor = TransientTerm() == (
            DiffusionTerm(coeff=1e-5) + evap + transpiration - cond_term
        )
        eq_temp = TransientTerm() == (
            DiffusionTerm(coeff=0.5) - P['Lv'] * cond_term - P['P_elt'] * ceiling_mask
        )
        eq_O2 = TransientTerm() == (
            DiffusionTerm(coeff=D_gas) + photosynthesis - respiration - P['O2_cons']
        )
        eq_CO2 = TransientTerm() == (
            DiffusionTerm(coeff=D_gas) - photosynthesis + respiration + P['CO2_prod']
        )

        # Solve
        eq_theta.solve(var=theta, dt=DT)
        eq_vapor.solve(var=Cv, dt=DT)
        eq_temp.solve(var=Temp, dt=DT)
        eq_O2.solve(var=O2, dt=DT)
        eq_CO2.solve(var=CO2, dt=DT)
        eq_S.solve(var=S, dt=DT)
        eq_M.solve(var=M, dt=DT)
        eq_bio.solve(var=biomass, dt=DT)

    # Trim arrays to actual recorded length
    for key in hist:
        hist[key] = hist[key][:idx]

    final_CO2 = hist['CO2'][-1]
    final_Temp = hist['Temp'][-1]

    return hist, final_CO2, final_Temp

# ---------------------------------------------------------------------
# 6. SWEEP LOOP
# ---------------------------------------------------------------------
results = []

log("\n🚀 Starting 1D sweep (speed = {:.0f})...".format(FIXED_SPEED))
for g in gamma_values:
    log(f"   → gamma = {g:.4f}, speed = {FIXED_SPEED:.1f}")
    start = time.time()
    hist, co2_final, temp_final = run_simulation(g, FIXED_SPEED)
    elapsed = time.time() - start
    log(f"       CO2_final = {co2_final:.6f}  |  Temp_final = {temp_final:.2f} K  (time: {elapsed:.1f} s)")
    results.append({
        'gamma': g,
        'CO2_final': co2_final,
        'Temp_final': temp_final,
        'elapsed': elapsed,
        'hist': hist
    })

log("\n✅ Sweep completed.")

# ---------------------------------------------------------------------
# 7. SAVE CSV
# ---------------------------------------------------------------------
# Summary CSV
summary_file = f"sweep_gamma_1D_speed{FIXED_SPEED:.0f}_summary.csv"
with open(summary_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gamma', 'CO2_final_30d', 'Temp_final_K', 'elapsed_s'])
    for r in results:
        writer.writerow([r['gamma'], r['CO2_final'], r['Temp_final'], r['elapsed']])

# Full time series CSV
series_file = f"sweep_gamma_1D_speed{FIXED_SPEED:.0f}_series.csv"
with open(series_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['gamma', 'time_h', 'CO2', 'Temp', 'theta', 'biomass', 'S_mean', 'M_mean', 'entropy']
    writer.writerow(header)
    for r in results:
        g = r['gamma']
        hist = r['hist']
        for i in range(len(hist['time_h'])):
            row = [g, hist['time_h'][i], hist['CO2'][i], hist['Temp'][i],
                   hist['theta'][i], hist['biomass'][i], hist['S_mean'][i],
                   hist['M_mean'][i], hist['entropy'][i]]
            writer.writerow(row)

log(f"📁 Summary CSV: {summary_file}")
log(f"📁 Full series CSV: {series_file}")

# ---------------------------------------------------------------------
# 8. FIGURES
# ---------------------------------------------------------------------
# 8.1 Bifurcation curve: gamma vs CO2_final
gammas = [r['gamma'] for r in results]
co2s = [r['CO2_final'] for r in results]

plt.figure(figsize=(8, 6))
plt.plot(gammas, co2s, 'o-', linewidth=2, markersize=8, color='darkblue')
plt.xlabel('γ (cooperation coefficient)', fontsize=12)
plt.ylabel('CO₂ final after 30 days (fraction)', fontsize=12)
plt.title(f'Bifurcation at signal speed = {FIXED_SPEED:.0f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
bifurcation_file = f"sweep_gamma_1D_speed{FIXED_SPEED:.0f}_bifurcation.png"
plt.savefig(bifurcation_file, dpi=300)
log(f"📊 Bifurcation figure: {bifurcation_file}")

# 8.2 Time series for each gamma (overlay plot)
plt.figure(figsize=(10, 6))
for r in results:
    hist = r['hist']
    g = r['gamma']
    plt.plot(hist['time_h'], hist['CO2'], label=f'γ={g:.3f}', linewidth=1.5)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('CO₂ (fraction)', fontsize=12)
plt.title(f'CO₂ evolution for various γ (speed={FIXED_SPEED:.0f})', fontsize=14)
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
timeseries_file = f"sweep_gamma_1D_speed{FIXED_SPEED:.0f}_timeseries.png"
plt.savefig(timeseries_file, dpi=300)
log(f"📊 Time series figure: {timeseries_file}")

# ---------------------------------------------------------------------
# 9. SUMMARY LOG
# ---------------------------------------------------------------------
log("\n" + "=" * 80)
log(f"SUMMARY OF RESULTS (speed = {FIXED_SPEED:.0f})")
log("=" * 80)
for r in results:
    log(f"γ = {r['gamma']:.4f}  →  CO₂_final = {r['CO2_final']:.6f}  |  Temp = {r['Temp_final']:.2f} K")
log("\n✅ All done.")
log("=" * 80)
log_handle.close()