import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid3D, TransientTerm, DiffusionTerm, ConvectionTerm
from build_masks import build_masks

nx = ny = nz = 17
dx = dy = dz = 0.05
mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)

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

geometry_params = {
    'x_split': 0.5, 'z_sol_continental': 0.4, 'z_sol_marsh': 0.2,
    'z_plant_max_continental': 0.8, 'z_plant_max_marsh': 0.6, 'z_ceil': 0.9,
}

gamma_fixed = 0.2
n_hill = 2; Km = 0.01; k_basal = 5e-5; lambda_S = 0.08; k_prod = 0.02
P_seuil = 1e-5; gamma_M = 0.08; delta_M = 0.02; M_max = 1.0
D_myco = 1e-4; D_M = 1e-6; chi = 1e-3

N_PERIODS = 20   # 20 périodes suffisent pour une estimation fiable

def run_period(forcing_period):
    STROBE_STEPS = int(forcing_period / dt)
    N_STEPS = N_PERIODS * STROBE_STEPS

    theta   = CellVariable(mesh=mesh, value=0.3,  hasOld=True)
    Temp    = CellVariable(mesh=mesh, value=298.0, hasOld=True)
    Cv      = CellVariable(mesh=mesh, value=0.02, hasOld=True)
    O2      = CellVariable(mesh=mesh, value=0.21, hasOld=True)
    CO2     = CellVariable(mesh=mesh, value=0.0004, hasOld=True)
    biomass = CellVariable(mesh=mesh, value=0.1,  hasOld=True)
    S = CellVariable(mesh=mesh, value=0.01, hasOld=True)
    M = CellVariable(mesh=mesh, value=0.1,  hasOld=True)

    M_ceil, M_photo, M_resp, M_transp, M_uptake = build_masks(mesh, geometry_params)
    soil_mask = M_uptake; ceiling_mask = M_ceil

    strobe_CO2 = []

    for step in range(N_STEPS):
        t = step * dt
        theta.updateOld(); Temp.updateOld(); Cv.updateOld()
        O2.updateOld(); CO2.updateOld(); biomass.updateOld()
        S.updateOld(); M.updateOld()

        day = 0.5 + 0.5 * np.sin(2 * np.pi * t / forcing_period)
        RH = float(np.mean(Cv.value) / 0.03)
        cond_val = max(0, k_cond * (RH - 0.6))
        cond_term = CellVariable(mesh=mesh, value=cond_val)

        peltier = P_elt; Lv_eff = 0.5
        evap = 0.001 * (1 + day) * theta * (Temp - 273.0)
        uptake = 0.02 * theta * soil_mask

        f_T = np.maximum(0, np.exp(-((Temp.value - T_opt) / T_range)**2))
        bio_active = np.maximum(0, biomass.value) * soil_mask
        Pmax_eff = Pmax * (1 + 0.1 * day)
        photosynthesis = Pmax_eff * (CO2 / (CO2 + k_CO2)) * f_T * theta * bio_active * M_photo
        resp_factor = np.maximum(0.1, 1 + 0.05 * (Temp.value - T_opt))
        respiration = 0.002 * bio_active * resp_factor * M_resp
        transpiration = transpiration_coeff * bio_active * (1 + day) * M_transp

        eq_theta = TransientTerm() == DiffusionTerm(coeff=1e-6) - uptake - transpiration + cond_term
        eq_vapor = TransientTerm() == DiffusionTerm(coeff=1e-5) + evap + transpiration - cond_term
        eq_temp  = TransientTerm() == DiffusionTerm(coeff=0.5) - Lv_eff * cond_term - peltier * ceiling_mask
        eq_O2  = TransientTerm() == DiffusionTerm(coeff=D_gas) + photosynthesis - respiration - astro_O2_cons
        eq_CO2 = TransientTerm() == DiffusionTerm(coeff=D_gas) - photosynthesis + respiration + astro_CO2_prod

        stress = np.maximum(0, P_seuil - photosynthesis.value)
        source_S = (k_prod * stress + k_basal) * soil_mask
        convection_S = ConvectionTerm(coeff=-chi * M.faceGrad, var=S)
        eq_S = (TransientTerm(var=S) == DiffusionTerm(coeff=D_myco, var=S) +
                convection_S + source_S - lambda_S * S)
        eq_M = (TransientTerm(var=M) == DiffusionTerm(coeff=D_M, var=M) +
                gamma_M * S * (1 - M / M_max) * soil_mask - delta_M * M)
        S_safe = np.maximum(0, S.value)
        hill_term = S_safe**n_hill / (Km**n_hill + S_safe**n_hill + 1e-8)
        cooperation_term = gamma_fixed * hill_term * biomass.value * soil_mask
        eq_bio = TransientTerm(var=biomass) == 0.6 * photosynthesis - respiration + cooperation_term

        eq_theta.solve(var=theta, dt=dt); eq_vapor.solve(var=Cv, dt=dt)
        eq_temp.solve(var=Temp, dt=dt); eq_O2.solve(var=O2, dt=dt)
        eq_CO2.solve(var=CO2, dt=dt); eq_S.solve(var=S, dt=dt)
        eq_M.solve(var=M, dt=dt); eq_bio.solve(var=biomass, dt=dt)

        if step % STROBE_STEPS == 0:
            strobe_CO2.append(np.mean(CO2.value))

    return np.std(strobe_CO2[-10:])

# Périodes déjà faites : 30. Récupérer sa valeur depuis le log
amplitude_30 = 0.003242
periods_remaining = [40, 50, 60, 70, 80]
amplitudes = [amplitude_30]

for per in periods_remaining:
    amp = run_period(per)
    amplitudes.append(amp)
    print(f"Period {per:.1f} s, amplitude = {amp:.6f}")

# Combiner toutes les périodes
all_periods = [30] + periods_remaining

plt.figure(figsize=(8, 5))
plt.plot(all_periods, amplitudes, 's-', linewidth=2, markersize=8, color='darkred')
plt.xlabel('Forcing period (s)')
plt.ylabel('Amplitude of CO₂ cycle (std dev)')
plt.title('Robustness of forced limit cycle to period changes')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_Period_Amplitude.png', dpi=300)

np.savetxt('period_sweep_complete.csv',
           np.column_stack([all_periods, amplitudes]),
           delimiter=',', header='period,amplitude', comments='')
print("Period sweep terminé. Données sauvegardées.")