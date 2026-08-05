#!/usr/bin/env python3
# =====================================================================
# ALGO_3_3D_BIO_V10.0 — RE‑DIMENSIONNALIZED IN REAL SECONDS
#
# Version: V10.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Based on:
#   - V6.2 (stroboscopic Poincaré, WWW, day/night forcing)
#   - V8.3 (dimensionless parameters)
#   - scale_params_V10.py (scaling)
#
# MODIFICATIONS MAJEURES:
#   1. Load parameters from a JSON file (params_real.json)
#   2. Real time scale: tau_base = 3600 s (1 hour)
#   3. Time step DT = DT_ADIM * tau_base = 360 s (6 min)
#   4. Default forcing period: 50 * tau_base = 180 000 s (50 h)
#   5. All biological rates are in s^-1
#   6. Preservation of the calculation structure (FiPy, masks, etc.)
# =====================================================================

import json
import numpy as np
from fipy import (
    CellVariable, Grid3D, TransientTerm, DiffusionTerm,
    ConvectionTerm
)
import matplotlib
matplotlib.use('Agg')  # For headless servers
import matplotlib.pyplot as plt
import csv
from build_masks import build_masks

# ---------------------------------------------------------------------
# 0. LOGGING (to follow with tail -f)
# ---------------------------------------------------------------------
LOG_FILE = "sim_v10_progress.log"
log_handle = open(LOG_FILE, "w", buffering=1)  # Line buffering

def log_message(msg):
    """Writes a message to the console AND to the log file."""
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log_message("=" * 60)
log_message("ALGO_3_3D_BIO_V10.0 - RE-DIMENSIONNALIZED SIMULATION")
log_message("=" * 60)

# ---------------------------------------------------------------------
# 1. LOAD PARAMETERS FROM JSON FILE
# ---------------------------------------------------------------------
try:
    with open('params_real.json', 'r') as f:
        P = json.load(f)
    log_message("✅ Parameters loaded from params_real.json")
except FileNotFoundError:
    log_message("❌ File params_real.json not found.")
    log_message("   Please generate it with scale_params_V10.py")
    exit(1)

# ---------------------------------------------------------------------
# 2. TIME SCALE (real seconds)
# ---------------------------------------------------------------------
TAU_BASE = 3600.0                 # 1 hour (base scale)
DT_ADIM = 0.1                     # Dimensionless time step (V6.2)
DT = DT_ADIM * TAU_BASE           # 360 seconds (6 minutes)

# Forcing period (day/night): we keep the V6.2 ratio (50)
FORCING_PERIOD_ADIM = 50.0
FORCING_PERIOD = FORCING_PERIOD_ADIM * TAU_BASE  # 180 000 s (50 h)

# Number of steps per stroboscopic period
STROBE_STEPS = int(FORCING_PERIOD / DT)           # 500 steps
N_PERIODS = 40                                    # 40 cycles
N_STEPS = N_PERIODS * STROBE_STEPS                # 20 000 steps
                                                    # = 7.2 × 10^6 s (~83 days)

log_message(f"⏱️  Real time scale:")
log_message(f"   DT = {DT:.1f} s (6 min)")
log_message(f"   FORCING_PERIOD = {FORCING_PERIOD:.1f} s ({FORCING_PERIOD/3600:.1f} h)")
log_message(f"   Number of periods = {N_PERIODS}")
log_message(f"   Total simulated: {N_STEPS * DT / 86400:.2f} days")

# ---------------------------------------------------------------------
# 3. 3D MESH (identical to V6.2)
# ---------------------------------------------------------------------
nx = ny = nz = 17
dx = dy = dz = 0.05
mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

# ---------------------------------------------------------------------
# 4. PARAMETERS EXTRACTED FROM JSON (with default values)
# ---------------------------------------------------------------------
# Biological / transport rates (s^-1 unless stated)
Pmax                = P.get('Pmax', 0.01)                 # s^-1
R_maint             = P.get('R_maint', 0.002)             # s^-1
k_CO2               = P.get('k_CO2', 0.05)               # dimensionless (half‑saturation)
transpiration_coeff = P.get('transpiration_coeff', 0.0005) # s^-1
evap_coeff          = P.get('evap_coeff', 0.001)          # s^-1
uptake_coeff        = P.get('uptake_coeff', 0.02)         # s^-1
k_cond              = P.get('k_cond', 0.5)               # s^-1
P_elt               = P.get('P_elt', 0.01)                # K/s (or W/m³)
Lv                  = P.get('Lv', 0.5)                   # dimensionless (latent heat)
gamma               = P.get('gamma', 0.2)                # dimensionless
gamma_M             = P.get('gamma_M', 0.05)              # s^-1
lambda_S            = P.get('lambda_S', 0.1)             # s^-1
k_prod              = P.get('k_prod', 0.01)               # s^-1
Km                  = P.get('Km', 0.01)                   # dimensionless (Hill half‑saturation)
k_basal             = P.get('k_basal', 1e-5)              # s^-1
P_seuil             = P.get('P_seuil', 1e-6)              # dimensionless (stress threshold)
Y_B                 = P.get('Y_B', 0.6)                  # s^-1 (yield)
delta_M             = P.get('delta_M', 0.01)              # s^-1
n_hill              = P.get('n_hill', 2)                # dimensionless (Hill exponent)
O2_cons             = P.get('O2_cons', 9.72e-9)          # s^-1 (crew consumption)
CO2_prod            = P.get('CO2_prod', 1.157e-8)        # s^-1 (crew production)
T_opt               = P.get('T_opt', 298.0)              # K (optimal temperature)
T_range             = P.get('T_range', 10.0)             # K (temperature tolerance)
alpha_heat          = P.get('alpha_heat', 0.5)           # dimensionless (radiative heat)
eta_heat            = P.get('eta_heat', 0.05)            # dimensionless (metabolic heat)

# Geometry (unchanged)
geometry_params = {
    'x_split': P['x_split'],
    'z_ceil': P['z_ceil'],
    'z_sol_continental': P['z_sol_continental'],
    'z_plant_max_continental': P['z_plant_max_continental'],
    'z_sol_marsh': P['z_sol_marsh'],
    'z_plant_max_marsh': P['z_plant_max_marsh'],
}

# Fixed physical parameters (consistent with V6.2)
D_gas   = 2e-5          # gas diffusivity (m²/s)
M_max   = 1.0           # maximum mycelium capacity (dimensionless)
D_myco  = 1e-4          # signal diffusivity (m²/s)
D_M     = 1e-6          # mycelium diffusivity (m²/s)
chi     = 1e-3          # signal convection coefficient (dimensionless)

# ---------------------------------------------------------------------
# 5. SIMULATION FUNCTION
# ---------------------------------------------------------------------
def run_simulation(use_WWW=True):
    """Runs the 3D simulation with or without the Wood Wide Web."""
    log_message(f"   -> Starting simulation (WWW = {use_WWW})...")

    # State variables
    theta   = CellVariable(mesh=mesh, value=0.3,  hasOld=True)
    Temp    = CellVariable(mesh=mesh, value=298.0, hasOld=True)
    Cv      = CellVariable(mesh=mesh, value=0.02, hasOld=True)
    O2      = CellVariable(mesh=mesh, value=0.21, hasOld=True)
    CO2     = CellVariable(mesh=mesh, value=0.0004, hasOld=True)
    biomass = CellVariable(mesh=mesh, value=0.1,  hasOld=True)
    S       = CellVariable(mesh=mesh, value=0.01 if use_WWW else 0.0, hasOld=True)
    M       = CellVariable(mesh=mesh, value=0.1 if use_WWW else 0.0, hasOld=True)

    # Masks
    M_ceil, M_photo, M_resp, M_transp, M_uptake = build_masks(mesh, geometry_params)
    soil_mask = M_uptake
    ceiling_mask = M_ceil

    # Storage
    strobe_WL = []
    strobe_WR = []
    strobe_CO2 = []
    strobe_theta = []
    theta_hist, CO2_hist, Temp_hist = [], [], []

    # Time loop
    for step in range(N_STEPS):
        t = step * DT

        if step % 500 == 0:
            log_message(f"   Step {step}/{N_STEPS}  |  t = {t/3600:.1f} h")

        # Update old values
        theta.updateOld(); Temp.updateOld(); Cv.updateOld()
        O2.updateOld(); CO2.updateOld(); biomass.updateOld()
        if use_WWW:
            S.updateOld(); M.updateOld()

        # Day/night forcing (real period)
        day = 0.5 + 0.5 * np.sin(2 * np.pi * t / FORCING_PERIOD)

        # Condensation
        RH = float(np.mean(Cv.value) / 0.03)
        cond_val = max(0, k_cond * (RH - 0.6))
        cond_term = CellVariable(mesh=mesh, value=cond_val)

        # Evapotranspiration and uptake
        evap = evap_coeff * (1 + day) * theta * (Temp - 273.0)
        uptake = uptake_coeff * theta * soil_mask
        transpiration = transpiration_coeff * biomass * (1 + day) * M_transp

        # Photosynthetic response to temperature
        f_T = np.maximum(0, np.exp(-((Temp.value - T_opt) / T_range)**2))
        bio_active = np.maximum(0, biomass.value) * soil_mask

        # Photosynthesis (modulated by forcing)
        Pmax_eff = Pmax * (1 + 0.1 * day)
        photosynthesis = (
            Pmax_eff
            * (CO2 / (CO2 + k_CO2))
            * f_T
            * theta
            * bio_active
            * M_photo
        )

        # Respiration
        resp_factor = np.maximum(0.1, 1 + 0.05 * (Temp.value - T_opt))
        respiration = R_maint * bio_active * resp_factor * M_resp

        # --- PDEs ---
        eq_theta = TransientTerm() == (
            DiffusionTerm(coeff=1e-6)
            - uptake
            - evap
            + cond_term
        )
        eq_vapor = TransientTerm() == (
            DiffusionTerm(coeff=1e-5)
            + evap
            + transpiration
            - cond_term
        )
        eq_temp = TransientTerm() == (
            DiffusionTerm(coeff=0.5)
            - Lv * cond_term
            - P_elt * ceiling_mask
        )
        eq_O2 = TransientTerm() == (
            DiffusionTerm(coeff=D_gas)
            + photosynthesis
            - respiration
            - O2_cons
        )
        eq_CO2 = TransientTerm() == (
            DiffusionTerm(coeff=D_gas)
            - photosynthesis
            + respiration
            + CO2_prod
        )

        if use_WWW:
            # Mycorrhizal signal
            stress = np.maximum(0, P_seuil - photosynthesis.value)
            source_S = (k_prod * stress + k_basal) * soil_mask
            convection_S = ConvectionTerm(coeff=-chi * M.faceGrad, var=S)

            eq_S = (
                TransientTerm(var=S)
                == DiffusionTerm(coeff=D_myco, var=S)
                + convection_S
                + source_S
                - lambda_S * S
            )

            # Mycelium
            eq_M = (
                TransientTerm(var=M)
                == DiffusionTerm(coeff=D_M, var=M)
                + gamma_M * S * (1 - M / M_max) * soil_mask
                - delta_M * M
            )

            # WWW cooperation (Hill term) — CORRECTED: divided by TAU_BASE
            S_safe = np.maximum(0, S.value)
            hill_term = S_safe**n_hill / (Km**n_hill + S_safe**n_hill + 1e-8)
            cooperation_term = (gamma * hill_term * biomass.value * soil_mask) / TAU_BASE

            eq_bio = TransientTerm(var=biomass) == (
                0.6 * photosynthesis
                - respiration
                + cooperation_term
            )
        else:
            eq_bio = TransientTerm(var=biomass) == (
                0.6 * photosynthesis
                - respiration
            )

        # Solve
        eq_theta.solve(var=theta, dt=DT)
        eq_vapor.solve(var=Cv, dt=DT)
        eq_temp.solve(var=Temp, dt=DT)
        eq_O2.solve(var=O2, dt=DT)
        eq_CO2.solve(var=CO2, dt=DT)
        if use_WWW:
            eq_S.solve(var=S, dt=DT)
            eq_M.solve(var=M, dt=DT)
        eq_bio.solve(var=biomass, dt=DT)

        # Save series (every 100 steps)
        if step % 100 == 0:
            theta_hist.append(np.mean(theta.value))
            CO2_hist.append(np.mean(CO2.value))
            Temp_hist.append(np.mean(Temp.value))

        # Stroboscopic sampling (exactly at each period)
        if step % STROBE_STEPS == 0:
            theta_3D = theta.value.reshape(nx, ny, nz)
            WL = np.mean(theta_3D[:nx//2, :, :]) * 100
            WR = np.mean(theta_3D[nx//2:, :, :]) * 100
            strobe_WL.append(WL)
            strobe_WR.append(WR)
            strobe_CO2.append(np.mean(CO2.value))
            strobe_theta.append(np.mean(theta.value))

    return (
        theta_hist, CO2_hist, Temp_hist,
        strobe_WL, strobe_WR, strobe_CO2, strobe_theta
    )

# ---------------------------------------------------------------------
# 6. EXECUTE BOTH SCENARIOS
# ---------------------------------------------------------------------
log_message("\n🚀 Starting simulation WITHOUT WWW...")
res_off = run_simulation(use_WWW=False)

log_message("\n🚀 Starting simulation WITH WWW...")
res_on = run_simulation(use_WWW=True)

# Unpack results
(
    theta_off, CO2_off, T_off,
    WL_off, WR_off, CO2strobe_off, theta_strobe_off
) = res_off
(
    theta_on, CO2_on, T_on,
    WL_on, WR_on, CO2strobe_on, theta_strobe_on
) = res_on

# ---------------------------------------------------------------------
# 7. FIGURES
# ---------------------------------------------------------------------
# 7.1 Time series (CO2 and T)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(CO2_off, label='Without WWW')
axs[0].plot(CO2_on, label='With WWW')
axs[0].set_title('CO₂ (volume fraction)')
axs[0].set_xlabel('Time steps (×100)')
axs[0].legend()
axs[1].plot(T_off, label='Without WWW')
axs[1].plot(T_on, label='With WWW')
axs[1].set_title('Temperature (K)')
axs[1].set_xlabel('Time steps (×100)')
axs[1].legend()
plt.tight_layout()
plt.savefig('Figure_V10_TimeSeries.png', dpi=300)
plt.close()

# 7.2 Stroboscopic Poincaré section (WL vs WR)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(WL_off, WR_off, 'o-', markersize=4, label='Without WWW')
ax.plot(WL_on, WR_on, 's-', markersize=4, label='With WWW')
ax.set_xlabel('W_L – Marsh biome (%)')
ax.set_ylabel('W_R – Continental biome (%)')
ax.set_title('Stroboscopic Poincaré section (period = {:.1f} h)'.format(FORCING_PERIOD/3600))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_V10_Poincare_Strobe.png', dpi=300)
plt.close()

# 7.3 Return maps (stroboscopic CO2)
def plot_return_map(values, title, filename):
    if len(values) > 2:
        vn = np.array(values[:-1])
        vnp1 = np.array(values[1:])
        plt.figure(figsize=(7, 7))
        plt.plot(vn, vnp1, 'o-', markersize=4)
        plt.plot([min(vn), max(vn)], [min(vn), max(vn)], 'k--')
        plt.xlabel('x_n')
        plt.ylabel('x_{n+1}')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

plot_return_map(
    CO2strobe_on,
    'Return map CO₂ – WITH WWW',
    'Figure_V10_ReturnMap_CO2_on.png'
)
plot_return_map(
    CO2strobe_off,
    'Return map CO₂ – WITHOUT WWW',
    'Figure_V10_ReturnMap_CO2_off.png'
)

# ---------------------------------------------------------------------
# 8. EXPORT STROBOSCOPIC DATA (CSV)
# ---------------------------------------------------------------------
with open('TECP_V10_stroboscopic.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'period', 'WL_off', 'WR_off', 'CO2_off', 'theta_off',
        'WL_on', 'WR_on', 'CO2_on', 'theta_on'
    ])
    for i in range(min(len(WL_off), len(WL_on))):
        writer.writerow([
            i,
            WL_off[i] if i < len(WL_off) else '',
            WR_off[i] if i < len(WR_off) else '',
            CO2strobe_off[i] if i < len(CO2strobe_off) else '',
            theta_strobe_off[i] if i < len(theta_strobe_off) else '',
            WL_on[i] if i < len(WL_on) else '',
            WR_on[i] if i < len(WR_on) else '',
            CO2strobe_on[i] if i < len(CO2strobe_on) else '',
            theta_strobe_on[i] if i < len(theta_strobe_on) else ''
        ])

log_message("\n✅ Stroboscopic data exported to TECP_V10_stroboscopic.csv")
log_message("✅ Figures generated.")
log_message("=" * 60)
log_message("SIMULATION V10.0 COMPLETED.")
log_message("=" * 60)

log_handle.close()