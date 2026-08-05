#!/usr/bin/env python3
# =====================================================================
# SWEEP_2D_V10 — TWO‑DIMENSIONAL SWEEP (γ × SIGNAL SPEED) FOR 30 DAYS
# 
# Description:
#   Performs a 2D parameter sweep over the cooperation coefficient γ
#   and the mycorrhizal signal speed. Each simulation runs for 30 days
#   with 24‑h forcing. Produces a phase diagram (heatmap), entropy
#   evolution, water cycle, and day/night highlight.
#
# Version: V10.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. γ values: [0.010, 0.013, 0.016, 0.020, 0.025].
#   2. Signal speeds: [4, 8, 16, 32, 64, 128, 168].
#   3. 30‑day simulations with full WWW dynamics.
#   4. Heatmap of CO₂ final (Figure 8).
#   5. Entropy, water cycle, and day/night highlight figures.
#   6. Saves full time series to sweep_2D_full.csv.
# =====================================================================

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
from mpl_toolkits.mplot3d import Axes3D
from build_masks import build_masks

# -----------------------------------------------------------------------------
# 1. LOGGING
# -----------------------------------------------------------------------------
LOG_FILE = "sweep_2D.log"
log_handle = open(LOG_FILE, "w", buffering=1)

def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("=" * 80)
log("SWEEP 2D V10 — PHASE DIAGRAM (gamma × signal_speed)")
log("=" * 80)

# -----------------------------------------------------------------------------
# 2. FIXED PARAMETERS (30 days, 24h forcing)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. LOAD BASE PARAMETERS (from params_real.json)
# -----------------------------------------------------------------------------
try:
    with open('params_real.json', 'r') as f:
        P = json.load(f)
    log("✅ Base parameters loaded from params_real.json")
except FileNotFoundError:
    log("❌ params_real.json not found.")
    exit(1)

# -----------------------------------------------------------------------------
# 4. SWEEP GRIDS
# -----------------------------------------------------------------------------
gamma_values = [0.010, 0.013, 0.016, 0.020, 0.025]   # as before
# signal_speed: multiplicative factor applied to k_prod, lambda_S, k_basal
# 1.0 corresponds to the value in params_real.json (which was derived from scale=168)
# We will test speeds from fast (4x) to slow (168x) to cover both regimes
signal_speeds = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 168.0]

log(f"Gamma values: {gamma_values}")
log(f"Signal speeds: {signal_speeds}")

# -----------------------------------------------------------------------------
# 5. SIMULATION FUNCTION (returns full history)
# -----------------------------------------------------------------------------
def run_simulation(gamma_val, speed_factor):
    """
    Run a 30‑day simulation for given gamma and signal speed factor.
    Returns a dict with all time series.
    """
    # Update gamma
    P['gamma'] = gamma_val

    # Apply signal speed factor to the mycorrhizal signal parameters
    # We keep base values from params_real.json and multiply by speed_factor
    # (speed_factor = 4 means faster signal, so we multiply the rates)
    # Actually, in our scaling, larger speed_factor means slower signal (since it's a scale).
    # To avoid confusion: speed_factor = 4 means we want the signal to be 4 times faster than baseline.
    # In the scaled parameters, the rates are divided by (TAU_BASE * s_i). So to make it faster,
    # we need to multiply the rates by speed_factor (or divide the s_i).
    # Simpler: we redefine the effective rates inside the simulation:
    # lambda_S_eff = lambda_S * speed_factor / 168.0? Wait, better: we keep the logic clear.
    # The original lambda_S in params_real.json was computed with s_myc_sig = 168.
    # If we want a speed factor of 4, we want the signal to be 168/4 = 42 times faster.
    # So we can set an effective s_myc_sig = 168.0 / speed_factor.
    # Therefore, we recompute the rates on the fly.
    # We'll extract the adimensional values from P? But P already contains scaled values.
    # Simpler: we directly recompute the rates using the base adim values from params_adim.
    # Since we don't have the adim file here, we'll use the current P values and adjust them.
    # For reproducibility, we'll multiply the rates by (168.0 / speed_factor) for the signal parameters.
    # That makes the signal faster when speed_factor > 1.
    # We'll apply this to lambda_S, k_prod, k_basal (and optionally gamma_M? no, gamma_M is growth).
    # For speed_factor = 168, we keep as is.
    # For speed_factor = 4, we multiply by 168/4 = 42 => signal much faster.
    scale_ratio = BASELINE_SPEED / speed_factor
    lambda_S_eff = P['lambda_S'] * scale_ratio
    k_prod_eff = P['k_prod'] * scale_ratio
    k_basal_eff = P['k_basal'] * scale_ratio
    # We keep gamma_M, delta_M unchanged (growth of mycelium remains slow, as it should)

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

    # Storage for time series (saved every 20 steps = every 2 hours)
    record_step = 20
    n_records = N_STEPS // record_step + 1
    hist = {
        'step': np.zeros(n_records, dtype=int),
        'time_h': np.zeros(n_records),
        'theta': np.zeros(n_records),
        'Cv': np.zeros(n_records),
        'Temp': np.zeros(n_records),
        'O2': np.zeros(n_records),
        'CO2': np.zeros(n_records),
        'biomass': np.zeros(n_records),
        'S_mean': np.zeros(n_records),
        'M_mean': np.zeros(n_records),
        'entropy': np.zeros(n_records),
        'WL': np.zeros(n_records),
        'WR': np.zeros(n_records),
        'cond_eff': np.zeros(n_records),
        'light_flux': np.zeros(n_records),   # fraction of max light
    }

    idx = 0
    for step in range(N_STEPS):
        t = step * DT
        if step % record_step == 0:
            hist['step'][idx] = step
            hist['time_h'][idx] = t / 3600.0
            # compute metrics
            theta_3D = theta.value.reshape(nx, ny, nz)
            hist['theta'][idx] = np.mean(theta.value)
            hist['Cv'][idx] = np.mean(Cv.value)
            hist['Temp'][idx] = np.mean(Temp.value)
            hist['O2'][idx] = np.mean(O2.value)
            hist['CO2'][idx] = np.mean(CO2.value)
            hist['biomass'][idx] = np.mean(biomass.value)
            hist['S_mean'][idx] = np.mean(S.value)
            hist['M_mean'][idx] = np.mean(M.value)
            # Entropy production (temperature gradient)
            T_3D = Temp.value.reshape(nx, ny, nz)
            grad_T = np.gradient(T_3D)
            grad_norm2 = sum(g**2 for g in grad_T)
            entropy = np.mean(grad_norm2 / (T_3D**2 + 1e-8))
            hist['entropy'][idx] = entropy
            # Left/right moisture
            WL = np.mean(theta_3D[:nx//2, :, :]) * 100
            WR = np.mean(theta_3D[nx//2:, :, :]) * 100
            hist['WL'][idx] = WL
            hist['WR'][idx] = WR
            # Condensation efficiency
            RH = float(np.mean(Cv.value) / 0.03)
            cond_val = max(0, P['k_cond'] * (RH - 0.6))
            Q_cond = P['Lv'] * cond_val
            Q_peltier = P['P_elt'] * np.mean(ceiling_mask.value)
            cond_eff = Q_cond / (Q_peltier + 1e-8)
            hist['cond_eff'][idx] = cond_eff
            # Light flux (day/night)
            day = 0.5 + 0.5 * np.sin(2 * np.pi * t / FORCING_PERIOD)
            hist['light_flux'][idx] = day
            idx += 1

        # --- Update old values ---
        theta.updateOld(); Temp.updateOld(); Cv.updateOld()
        O2.updateOld(); CO2.updateOld(); biomass.updateOld()
        S.updateOld(); M.updateOld()

        # --- Forcing ---
        day = 0.5 + 0.5 * np.sin(2 * np.pi * t / FORCING_PERIOD)

        # --- Condensation ---
        RH = float(np.mean(Cv.value) / 0.03)
        cond_val = max(0, P['k_cond'] * (RH - 0.6))
        cond_term = CellVariable(mesh=mesh, value=cond_val)

        # --- Evapotranspiration ---
        evap = P['evap_coeff'] * (1 + day) * theta * (Temp - 273.0)
        uptake = P['uptake_coeff'] * theta * soil_mask
        transpiration = P['transpiration_coeff'] * biomass * (1 + day) * M_transp

        # --- Photosynthesis ---
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

        # --- Respiration ---
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

        # --- Physical PDEs ---
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

        # --- Solve ---
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
    return hist

# -----------------------------------------------------------------------------
# 6. SWEEP LOOP (gamma × speed)
# -----------------------------------------------------------------------------
all_results = []   # list of dicts {gamma, speed, CO2_final, ...}
log("\n🚀 Starting 2D sweep...")

for g in gamma_values:
    for sp in signal_speeds:
        log(f"   → gamma = {g:.4f}, speed = {sp:.1f}")
        start = time.time()
        hist = run_simulation(g, sp)
        elapsed = time.time() - start
        final_CO2 = hist['CO2'][-1]
        log(f"       CO2_final = {final_CO2:.6f}  (time: {elapsed:.1f} s)")
        # Store summary
        all_results.append({
            'gamma': g,
            'speed': sp,
            'CO2_final': final_CO2,
            'elapsed': elapsed,
            'hist': hist   # keep full history for later plotting
        })

log("\n✅ 2D sweep completed.")

# -----------------------------------------------------------------------------
# 7. SAVE FULL CSV (all variables for each simulation)
# -----------------------------------------------------------------------------
csv_file = "sweep_2D_full.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header
    header = ['gamma', 'speed', 'step', 'time_h', 'theta', 'Cv', 'Temp',
              'O2', 'CO2', 'biomass', 'S_mean', 'M_mean', 'entropy',
              'WL', 'WR', 'cond_eff', 'light_flux']
    writer.writerow(header)
    for res in all_results:
        g = res['gamma']
        sp = res['speed']
        hist = res['hist']
        n = len(hist['step'])
        for i in range(n):
            row = [g, sp, hist['step'][i], hist['time_h'][i],
                   hist['theta'][i], hist['Cv'][i], hist['Temp'][i],
                   hist['O2'][i], hist['CO2'][i], hist['biomass'][i],
                   hist['S_mean'][i], hist['M_mean'][i], hist['entropy'][i],
                   hist['WL'][i], hist['WR'][i], hist['cond_eff'][i],
                   hist['light_flux'][i]]
            writer.writerow(row)
log(f"📁 Full CSV saved: {csv_file}")

# -----------------------------------------------------------------------------
# 8. HEATMAP: CO2_final vs gamma vs speed (log scale for speed)
# -----------------------------------------------------------------------------
# Build matrix
gamma_unique = sorted(set(r['gamma'] for r in all_results))
speed_unique = sorted(set(r['speed'] for r in all_results))
Z = np.zeros((len(speed_unique), len(gamma_unique)))
for i, sp in enumerate(speed_unique):
    for j, g in enumerate(gamma_unique):
        for res in all_results:
            if res['gamma'] == g and res['speed'] == sp:
                Z[i, j] = res['CO2_final']
                break

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(Z, aspect='auto', cmap='viridis',
               extent=[gamma_unique[0], gamma_unique[-1],
                       np.log10(speed_unique[0]), np.log10(speed_unique[-1])],
               origin='lower')
ax.set_xlabel('γ (cooperation coefficient)')
ax.set_ylabel('log10(signal speed factor)')
ax.set_title('CO₂ final after 30 days – Phase diagram')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('CO₂ (fraction)')
plt.tight_layout()
plt.savefig('heatmap_phase.png', dpi=300)
log("📊 Heatmap saved: heatmap_phase.png")

# -----------------------------------------------------------------------------
# 9. ENTROPY EVOLUTION for a representative case (e.g., gamma=0.016, speed=4)
# -----------------------------------------------------------------------------
# Find the case
rep_case = None
for res in all_results:
    if res['gamma'] == 0.016 and res['speed'] == 4.0:
        rep_case = res
        break
if rep_case is not None:
    hist = rep_case['hist']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist['time_h'], hist['entropy'], 'b-', linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Entropy production (a.u.)')
    ax.set_title(f'Entropy evolution (γ={rep_case["gamma"]:.3f}, speed={rep_case["speed"]:.1f})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('entropy_evolution.png', dpi=300)
    log("📊 Entropy figure saved: entropy_evolution.png")
else:
    log("⚠️ Representative case for entropy not found (gamma=0.016, speed=4).")

# -----------------------------------------------------------------------------
# 10. WATER CYCLE: Liquid, vapor, condensation efficiency (for same case)
# -----------------------------------------------------------------------------
if rep_case is not None:
    hist = rep_case['hist']
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Liquid (theta) and vapor (Cv)
    ax1.plot(hist['time_h'], hist['theta'] * 100, 'b-', label='Soil moisture θ (%)')
    ax1.plot(hist['time_h'], hist['Cv'] * 1000, 'g-', label='Vapor Cv ×1000')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Fraction (%)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hist['time_h'], hist['cond_eff'], 'r-', linewidth=1.5, label='Condensation efficiency')
    ax2.set_ylabel('Condensation efficiency')
    ax2.legend(loc='upper right')
    plt.title(f'Water cycle (γ={rep_case["gamma"]:.3f}, speed={rep_case["speed"]:.1f})')
    plt.tight_layout()
    plt.savefig('water_cycle.png', dpi=300)
    log("📊 Water cycle figure saved: water_cycle.png")
else:
    log("⚠️ Representative case for water cycle not found.")

# -----------------------------------------------------------------------------
# 11. DAY/NIGHT HIGHLIGHT with light flux (for same case)
# -----------------------------------------------------------------------------
if rep_case is not None:
    hist = rep_case['hist']
    # We'll plot CO2 and temperature with light flux shading
    fig, ax = plt.subplots(figsize=(12, 6))
    # Shade day/night
    light = hist['light_flux']
    time = hist['time_h']
    ax.fill_between(time, 0, 1, where=(light > 0.5), color='yellow', alpha=0.3, label='Day')
    ax.fill_between(time, 0, 1, where=(light <= 0.5), color='navy', alpha=0.15, label='Night')
    # Plot CO2 and temperature
    ax.plot(time, hist['CO2'] * 1000, 'b-', linewidth=2, label='CO₂ (ppm)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('CO₂ (ppm)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.plot(time, hist['Temp'], 'r-', linewidth=1.5, label='Temperature (K)')
    ax2.set_ylabel('Temperature (K)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')

    # Add light flux info
    max_light = np.max(light)
    min_light = np.min(light)
    ax.text(0.02, 0.95, f'Light flux amplitude: {max_light - min_light:.2f} (rel.)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title(f'Day/night cycle – γ={rep_case["gamma"]:.3f}, speed={rep_case["speed"]:.1f}')
    plt.tight_layout()
    plt.savefig('daynight_highlight.png', dpi=300)
    log("📊 Day/night highlight saved: daynight_highlight.png")
else:
    log("⚠️ Representative case for day/night not found.")

# -----------------------------------------------------------------------------
# 12. SUMMARY LOG
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("SUMMARY OF RESULTS")
log("=" * 80)
for res in all_results:
    log(f"γ = {res['gamma']:.4f}, speed = {res['speed']:.1f} → CO₂_final = {res['CO2_final']:.6f}")
log("\n✅ All done. Figures and CSV saved.")
log("=" * 80)
log_handle.close()