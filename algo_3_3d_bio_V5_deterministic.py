# =====================================================================
# ALGO_3_3D_BIO_V5 — DETERMINISTIC 3D SIMULATION WITHOUT MYCORRHIZAL NETWORK
# 
# Description:
#   3D reaction‑diffusion simulation of the TECP reactor without the
#   Wood Wide Web (WWW). The system degrades slowly due to water loss
#   and CO₂ accumulation. Thermodynamic metrics (entropy, condensation
#   efficiency, energy balance) are computed.
#
# Version: V5
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. 3D mesh (17×17×17) with Boolean masks.
#   2. Deterministic simulation for 5000 steps (500 s).
#   3. Computes soil moisture, vapour, temperature, O₂, CO₂, biomass.
#   4. Thermodynamic metrics: entropy production, condensation efficiency.
#   5. Generates Figures 3 and 4 (biological state and thermodynamic metrics).
# =====================================================================

import numpy as np
from fipy import CellVariable, Grid3D, TransientTerm, DiffusionTerm
import matplotlib
#matplotlib.use('TkAgg')  # Use 'Agg' on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from build_masks import build_masks

# -------------------------------
# 1. Mesh
# -------------------------------
nx = ny = nz = 17
dx = dy = dz = 0.05
mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

# -------------------------------
# 2. State Variables
# -------------------------------
theta   = CellVariable(mesh=mesh, value=0.3,  hasOld=True)
Temp    = CellVariable(mesh=mesh, value=298.0, hasOld=True)
Cv      = CellVariable(mesh=mesh, value=0.02, hasOld=True)
O2      = CellVariable(mesh=mesh, value=0.21, hasOld=True)
CO2     = CellVariable(mesh=mesh, value=0.0004, hasOld=True)
biomass = CellVariable(mesh=mesh, value=0.1,  hasOld=True)

# -------------------------------
# 3. Physical and Biological Parameters
# -------------------------------
dt = 0.1
D_gas = 2e-5
scale_astro = 0.001
astro_O2_cons = 0.84 / 86400 * scale_astro
astro_CO2_prod = 1.0 / 86400 * scale_astro

Pmax = 0.01
k_CO2 = 0.05
T_opt = 298.0
T_range = 10.0
k_cond = 0.5
P_elt = 0.05
transpiration_coeff = 0.0005

# -------------------------------
# 4. Geometry Configuration (Two Biomes)
# -------------------------------
geometry_params = {
    'x_split': 0.5,
    'z_sol_continental': 0.4,
    'z_sol_marsh': 0.2,
    'z_plant_max_continental': 0.8,
    'z_plant_max_marsh': 0.6,
    'z_ceil': 0.9,
}

# Build masks
M_ceil, M_photo, M_resp, M_transp, M_uptake = build_masks(mesh, geometry_params)
soil_mask = M_uptake
ceiling_mask = M_ceil

# -------------------------------
# 5. Time Series Storage
# -------------------------------
theta_hist, Cv_hist, Temp_hist = [], [], []
O2_hist, CO2_hist, Biom_hist = [], [], []
W_L_hist, W_R_hist = [], []
entropy_hist, efficiency_hist, energy_hist = [], [], []

# -------------------------------
# 6. Deterministic Simulation (5000 steps)
# -------------------------------
N_STEPS = 5000

for step in range(N_STEPS):
    t = step * dt
    if step % 250 == 0:
        print(f"Step {step}/{N_STEPS} completed", flush=True)
    theta.updateOld()
    Temp.updateOld()
    Cv.updateOld()
    O2.updateOld()
    CO2.updateOld()
    biomass.updateOld()

    # --- Deterministic drivers (no noise) ---
    noise = 0.0
    day = 0.5 + 0.5 * np.sin(2 * np.pi * t / 50)  # day/night cycle

    # Condensation (deterministic)
    RH = float(np.mean(Cv.value) / 0.03)
    cond_val = max(0, k_cond * (RH - 0.6))
    cond_term = CellVariable(mesh=mesh, value=cond_val)

    peltier = P_elt
    Lv_eff = 0.5

    evap = 0.001 * (1 + day) * theta * (Temp - 273.0)
    uptake = 0.02 * theta * soil_mask

    # --- Biological coupling ---
    f_T = np.maximum(0, np.exp(-((Temp.value - T_opt) / T_range)**2))
    bio_active = np.maximum(0, biomass.value) * soil_mask

    Pmax_eff = Pmax * (1 + 0.1 * day)
    photosynthesis = Pmax_eff * (CO2 / (CO2 + k_CO2)) * f_T * theta * bio_active * M_photo

    resp_factor = np.maximum(0.1, 1 + 0.05 * (Temp.value - T_opt))
    respiration = 0.002 * bio_active * resp_factor * M_resp

    transpiration = transpiration_coeff * bio_active * (1 + day) * M_transp

    # --- PDE solving ---
    eq_theta = TransientTerm() == DiffusionTerm(coeff=1e-6) - uptake - evap + cond_term
    eq_vapor = TransientTerm() == DiffusionTerm(coeff=1e-5) + evap + transpiration - cond_term
    eq_temp  = TransientTerm() == DiffusionTerm(coeff=0.5) - Lv_eff * cond_term - peltier * ceiling_mask

    eq_O2  = TransientTerm() == DiffusionTerm(coeff=D_gas) + photosynthesis - respiration - astro_O2_cons
    eq_CO2 = TransientTerm() == DiffusionTerm(coeff=D_gas) - photosynthesis + respiration + astro_CO2_prod
    eq_bio = TransientTerm() == 0.6 * photosynthesis - respiration

    eq_theta.solve(var=theta, dt=dt)
    eq_vapor.solve(var=Cv, dt=dt)
    eq_temp.solve(var=Temp, dt=dt)
    eq_O2.solve(var=O2, dt=dt)
    eq_CO2.solve(var=CO2, dt=dt)
    eq_bio.solve(var=biomass, dt=dt)

    # --- Post-processing & metrics ---
    Temp_3D = Temp.value.reshape(nx, ny, nz)
    theta_3D = theta.value.reshape(nx, ny, nz)

    # 1. Entropy production
    grad_T = np.gradient(Temp_3D)
    grad_norm_T = np.sqrt(sum(g**2 for g in grad_T))
    entropy_prod = np.mean((grad_norm_T**2) / (Temp_3D**2 + 1e-8))
    entropy_hist.append(entropy_prod)

    # 2. Energy metrics
    Q_cond = Lv_eff * cond_val
    Q_peltier_mean = peltier * np.mean(ceiling_mask.value)
    efficiency = Q_cond / (Q_peltier_mean + 1e-8)
    energy_balance = Q_peltier_mean - Q_cond
    efficiency_hist.append(efficiency)
    energy_hist.append(energy_balance)

    # 3. Left/Right moisture split
    W_L = np.mean(theta_3D[:nx//2, :, :]) * 100
    W_R = np.mean(theta_3D[nx//2:, :, :]) * 100
    W_L_hist.append(W_L)
    W_R_hist.append(W_R)

    # Standard metrics
    theta_hist.append(np.mean(theta.value))
    Cv_hist.append(np.mean(Cv.value))
    Temp_hist.append(np.mean(Temp.value))
    O2_hist.append(np.mean(O2.value))
    CO2_hist.append(np.mean(CO2.value))
    Biom_hist.append(np.mean(biomass.value))

print()

# -------------------------------
# 7. Figure 1 — Biological State Variables
# -------------------------------
fig1, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0, 0].plot(theta_hist, color='blue')
axs[0, 0].set_title("Soil moisture (θ)")
axs[0, 1].plot(Cv_hist, color='cyan')
axs[0, 1].set_title("Water vapour (Cv)")
axs[0, 2].plot(Temp_hist, color='red')
axs[0, 2].set_title("Temperature (K)")
axs[1, 0].plot(O2_hist, color='green')
axs[1, 0].set_title("Oxygen (O2)")
axs[1, 1].plot(CO2_hist, color='orange')
axs[1, 1].set_title("Carbon dioxide (CO2)")
axs[1, 2].plot(Biom_hist, color='darkgreen')
axs[1, 2].set_title("Biomass (B)")
for ax in axs.flat:
    ax.set_xlabel("Time steps")
plt.suptitle("TECP – Deterministic 3D Biological State (5000 steps)", fontsize=14)
plt.tight_layout()
plt.savefig("Figure4_Biology_Deterministic.png", dpi=300)
plt.savefig("Figure4_Biology_Deterministic.svg", format='svg')
plt.close()

# -------------------------------
# 8. Figure 2 — Thermodynamic Metrics and Phase Diagram
# -------------------------------
fig2, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(W_L_hist, W_R_hist, color='purple', marker='o',
               markersize=2, linestyle='', alpha=0.7)
axs[0, 0].set_xlabel("W_L – Marsh side (%)")
axs[0, 0].set_ylabel("W_R – Continental side (%)")
axs[0, 0].set_title("Phase Diagram (Deterministic)")
axs[0, 0].grid(True)

axs[0, 1].plot(entropy_hist, color='darkred')
axs[0, 1].set_title("Entropy Production (Dissipation)")
axs[0, 1].set_xlabel("Time steps")

axs[1, 0].plot(efficiency_hist, color='teal')
axs[1, 0].set_title("Condensation Efficiency η")
axs[1, 0].set_xlabel("Time steps")

axs[1, 1].plot(energy_hist, color='gray')
axs[1, 1].axhline(0, color='black', linestyle='--')
axs[1, 1].set_title("Net Energy Balance (Q_peltier – Q_cond)")
axs[1, 1].set_xlabel("Time steps")

for ax in axs.flat:
    ax.grid(True, alpha=0.3)
plt.suptitle("TECP – Deterministic Thermodynamic Metrics (5000 steps)", fontsize=14)
plt.tight_layout()
plt.savefig("Figure5_Thermo_Deterministic.png", dpi=300)
plt.savefig("Figure5_Thermo_Deterministic.svg", format='svg')
plt.close()

# -------------------------------
# 9. Export to CSV
# -------------------------------
with open('TECP_3D_deterministic_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Time_step', 'theta_mean', 'Cv_mean', 'Temp_mean',
        'O2_mean', 'CO2_mean', 'Biomass_mean',
        'W_L_pct', 'W_R_pct', 'Entropy_prod',
        'Cond_efficiency', 'Net_energy_balance'
    ])
    for i in range(len(theta_hist)):
        writer.writerow([
            i,
            theta_hist[i], Cv_hist[i], Temp_hist[i],
            O2_hist[i], CO2_hist[i], Biom_hist[i],
            W_L_hist[i], W_R_hist[i], entropy_hist[i],
            efficiency_hist[i], energy_hist[i]
        ])
print("Data exported to TECP_3D_deterministic_results.csv")