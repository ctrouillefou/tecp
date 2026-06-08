# ===============================
# ALGO_3_3D_BIO V6.2 — STROBOSCOPIC POINCARÉ SECTION
# Forced by external day/night cycle (period 50 s), with and without WWW
# ===============================
import numpy as np
from fipy import CellVariable, Grid3D, TransientTerm, DiffusionTerm, ConvectionTerm
import matplotlib
matplotlib.use('TkAgg')
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
# 2. Parameters
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

# Geometry
geometry_params = {
    'x_split': 0.5,
    'z_sol_continental': 0.4,
    'z_sol_marsh': 0.2,
    'z_plant_max_continental': 0.8,
    'z_plant_max_marsh': 0.6,
    'z_ceil': 0.9,
}

# --- WWW parameters (same as V6.1) ---
gamma_val = 0.2
n_hill = 2
Km = 0.01
k_basal = 5e-5
lambda_S = 0.08
k_prod = 0.02
P_seuil = 1e-5
gamma_M = 0.08
delta_M = 0.02
M_max = 1.0
D_myco = 1e-4
D_M = 1e-6
chi = 1e-3

# Forcing period (day/night cycle) in seconds
FORCING_PERIOD = 50.0   # as used in day modulation
STROBE_STEPS = int(FORCING_PERIOD / dt)   # 500 steps per period

# Total steps: 40 periods = 2000 s
N_PERIODS = 40
N_STEPS = N_PERIODS * STROBE_STEPS

# -------------------------------
# 3. Helper: one simulation run
# -------------------------------
def run_simulation(use_WWW=True):
    # Create mesh and variables
    theta   = CellVariable(mesh=mesh, value=0.3,  hasOld=True)
    Temp    = CellVariable(mesh=mesh, value=298.0, hasOld=True)
    Cv      = CellVariable(mesh=mesh, value=0.02, hasOld=True)
    O2      = CellVariable(mesh=mesh, value=0.21, hasOld=True)
    CO2     = CellVariable(mesh=mesh, value=0.0004, hasOld=True)
    biomass = CellVariable(mesh=mesh, value=0.1,  hasOld=True)
    S = CellVariable(mesh=mesh, value=0.01 if use_WWW else 0.0, hasOld=True)
    M = CellVariable(mesh=mesh, value=0.1 if use_WWW else 0.0, hasOld=True)

    M_ceil, M_photo, M_resp, M_transp, M_uptake = build_masks(mesh, geometry_params)
    soil_mask = M_uptake
    ceiling_mask = M_ceil

    # Storage for stroboscopic points
    strobe_WL = []
    strobe_WR = []
    strobe_CO2 = []
    strobe_theta = []

    # Storage for time series (reduced)
    theta_hist, CO2_hist, Temp_hist = [], [], []

    for step in range(N_STEPS):
        t = step * dt
        theta.updateOld(); Temp.updateOld(); Cv.updateOld()
        O2.updateOld(); CO2.updateOld(); biomass.updateOld()
        if use_WWW:
            S.updateOld(); M.updateOld()

        day = 0.5 + 0.5 * np.sin(2 * np.pi * t / FORCING_PERIOD)

        # Condensation
        RH = float(np.mean(Cv.value) / 0.03)
        cond_val = max(0, k_cond * (RH - 0.6))
        cond_term = CellVariable(mesh=mesh, value=cond_val)

        peltier = P_elt
        Lv_eff = 0.5

        evap = 0.001 * (1 + day) * theta * (Temp - 273.0)
        uptake = 0.02 * theta * soil_mask

        f_T = np.maximum(0, np.exp(-((Temp.value - T_opt) / T_range)**2))
        bio_active = np.maximum(0, biomass.value) * soil_mask

        Pmax_eff = Pmax * (1 + 0.1 * day)
        photosynthesis = Pmax_eff * (CO2 / (CO2 + k_CO2)) * f_T * theta * bio_active * M_photo

        resp_factor = np.maximum(0.1, 1 + 0.05 * (Temp.value - T_opt))
        respiration = 0.002 * bio_active * resp_factor * M_resp

        transpiration = transpiration_coeff * bio_active * (1 + day) * M_transp

        # PDEs
        eq_theta = TransientTerm() == DiffusionTerm(coeff=1e-6) - uptake - transpiration + cond_term
        eq_vapor = TransientTerm() == DiffusionTerm(coeff=1e-5) + evap + transpiration - cond_term
        eq_temp  = TransientTerm() == DiffusionTerm(coeff=0.5) - Lv_eff * cond_term - peltier * ceiling_mask

        eq_O2  = TransientTerm() == DiffusionTerm(coeff=D_gas) + photosynthesis - respiration - astro_O2_cons
        eq_CO2 = TransientTerm() == DiffusionTerm(coeff=D_gas) - photosynthesis + respiration + astro_CO2_prod

        if use_WWW:
            stress = np.maximum(0, P_seuil - photosynthesis.value)
            source_S = (k_prod * stress + k_basal) * soil_mask
            convection_S = ConvectionTerm(coeff=-chi * M.faceGrad, var=S)

            eq_S = (TransientTerm(var=S) ==
                    DiffusionTerm(coeff=D_myco, var=S) +
                    convection_S +
                    source_S - lambda_S * S)

            eq_M = (TransientTerm(var=M) ==
                    DiffusionTerm(coeff=D_M, var=M) +
                    gamma_M * S * (1 - M / M_max) * soil_mask - delta_M * M)

            S_safe = np.maximum(0, S.value)
            hill_term = S_safe**n_hill / (Km**n_hill + S_safe**n_hill + 1e-8)
            cooperation_term = gamma_val * hill_term * biomass.value * soil_mask
            eq_bio = TransientTerm(var=biomass) == 0.6 * photosynthesis - respiration + cooperation_term
        else:
            eq_bio = TransientTerm(var=biomass) == 0.6 * photosynthesis - respiration

        eq_theta.solve(var=theta, dt=dt)
        eq_vapor.solve(var=Cv, dt=dt)
        eq_temp.solve(var=Temp, dt=dt)
        eq_O2.solve(var=O2, dt=dt)
        eq_CO2.solve(var=CO2, dt=dt)
        if use_WWW:
            eq_S.solve(var=S, dt=dt)
            eq_M.solve(var=M, dt=dt)
        eq_bio.solve(var=biomass, dt=dt)

        # Store basic time series every 100 steps to reduce memory
        if step % 100 == 0:
            theta_hist.append(np.mean(theta.value))
            CO2_hist.append(np.mean(CO2.value))
            Temp_hist.append(np.mean(Temp.value))

        # Stroboscopic sampling exactly at each forcing period
        if step % STROBE_STEPS == 0:
            theta_3D = theta.value.reshape(nx, ny, nz)
            WL = np.mean(theta_3D[:nx//2, :, :]) * 100
            WR = np.mean(theta_3D[nx//2:, :, :]) * 100
            strobe_WL.append(WL)
            strobe_WR.append(WR)
            strobe_CO2.append(np.mean(CO2.value))
            strobe_theta.append(np.mean(theta.value))

    return (theta_hist, CO2_hist, Temp_hist,
            strobe_WL, strobe_WR, strobe_CO2, strobe_theta)

# -------------------------------
# 4. Run both scenarios
# -------------------------------
print("Running without WWW...")
res_off = run_simulation(use_WWW=False)
print("Running with WWW...")
res_on = run_simulation(use_WWW=True)

(theta_off, CO2_off, T_off, WL_off, WR_off, CO2strobe_off, theta_strobe_off) = res_off
(theta_on, CO2_on, T_on, WL_on, WR_on, CO2strobe_on, theta_strobe_on) = res_on

# -------------------------------
# 5. Plotting
# -------------------------------
# Time series (CO2 and T)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(CO2_off, label='Without WWW')
axs[0].plot(CO2_on, label='With WWW')
axs[0].set_title('CO2 concentration')
axs[0].legend()
axs[1].plot(T_off, label='Without WWW')
axs[1].plot(T_on, label='With WWW')
axs[1].set_title('Temperature (K)')
axs[1].legend()
plt.tight_layout()
plt.savefig('Figure_V6.2_TimeSeries.png', dpi=300)
plt.close()

# Stroboscopic Poincaré section (WL vs WR)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(WL_off, WR_off, 'o-', markersize=4, label='Without WWW')
ax.plot(WL_on, WR_on, 's-', markersize=4, label='With WWW')
ax.set_xlabel('W_L – Marsh side (%)')
ax.set_ylabel('W_R – Continental side (%)')
ax.set_title('Stroboscopic Poincaré Section (period = 50 s)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_V6.2_Poincare_Strobe.png', dpi=300)
plt.show()

# Return map using CO2 stroboscopic values
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
        plt.show()

plot_return_map(CO2strobe_on, 'Return Map (CO2 stroboscopic) – With WWW', 'Figure_V6.2_ReturnMap_CO2_on.png')
plot_return_map(CO2strobe_off, 'Return Map (CO2 stroboscopic) – Without WWW', 'Figure_V6.2_ReturnMap_CO2_off.png')

# Save stroboscopic data
with open('TECP_V6.2_stroboscopic.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['period', 'WL_off', 'WR_off', 'CO2_off', 'theta_off',
                     'WL_on', 'WR_on', 'CO2_on', 'theta_on'])
    for i in range(min(len(WL_off), len(WL_on))):
        writer.writerow([i, WL_off[i], WR_off[i], CO2strobe_off[i], theta_strobe_off[i],
                         WL_on[i], WR_on[i], CO2strobe_on[i], theta_strobe_on[i]])
print("Stroboscopic data saved.")