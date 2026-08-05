# =====================================================================
# ALGO_2_3D_BIO_V8.3 — HOMOGENEOUS STEADY STATE AND BIFURCATION ANALYSIS
# 
# Description:
#   Computes the homogeneous steady states of the TECP system with
#   the Wood Wide Web (WWW) and performs a bifurcation sweep over the
#   cooperation coefficient γ. The Jacobian matrix and eigenvalues are
#   calculated to assess linear stability.
#
# Version: V8.3
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. Analytical computation of effective volume fractions.
#   2. Homogeneous steady state with two coexisting branches ("Off" and "On").
#   3. Jacobian matrix and eigenvalue analysis.
#   4. Bifurcation sweep of γ from -0.05 to 0.30.
#   5. Generates Figure 2 (bifurcation diagram).
# =====================================================================

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# ===============================
# 1. GEOMETRIC FUNCTIONS
# ===============================

def compute_volume_fractions(params):
    """
    Compute the effective volume fractions for each process
    from the dimensionless geometric parameters.

    Returns a dictionary: f_ceil, f_photo, f_resp, f_transp, f_uptake
    """
    x_split = params.get('x_split', 0.5)

    # Relative widths (sum = 1)
    w_continental = x_split
    w_marsh = 1.0 - x_split

    # Normalised thresholds for the continental biome
    z_sol_c = params.get('z_sol_continental', 0.4)
    z_max_c = params.get('z_plant_max_continental', 0.8)
    z_resp_c = (z_sol_c + z_max_c) / 2.0

    # Normalised thresholds for the marsh biome
    z_sol_m = params.get('z_sol_marsh', 0.2)
    z_max_m = params.get('z_plant_max_marsh', 0.6)
    z_resp_m = (z_sol_m + z_max_m) / 2.0

    # Volume fractions for each process
    f_photo  = w_continental * (z_resp_c - z_sol_c) + w_marsh * (z_resp_m - z_sol_m)
    f_resp   = w_continental * z_resp_c + w_marsh * z_resp_m
    f_transp = w_continental * (z_resp_c - z_sol_c) + w_marsh * (z_resp_m - z_sol_m)
    f_uptake = w_continental * z_sol_c + w_marsh * z_sol_m
    f_ceil   = 1.0 - params.get('z_ceil', 0.9)

    return {
        'f_ceil':   f_ceil,
        'f_photo':  f_photo,
        'f_resp':   f_resp,
        'f_transp': f_transp,
        'f_uptake': f_uptake
    }


# ===============================
# 2. HOMOGENEOUS STEADY STATE (8 VARIABLES)
# ===============================

def homogeneous_steady_state_www(params):
    """
    Homogeneous steady state with Wood Wide Web.
    Variables: theta, Cv, T, O2, CO2, B, S, M  (8 variables)
    """

    # --- Compute volume fractions once ---
    vol_frac = compute_volume_fractions(params)
    f_ceil   = vol_frac['f_ceil']
    f_photo  = vol_frac['f_photo']
    f_resp   = vol_frac['f_resp']
    f_transp = vol_frac['f_transp']
    f_uptake = vol_frac['f_uptake']

    # --- Physical parameters ---
    Lv = params.get('Lv', 0.5)
    k_cond = params.get('k_cond', 0.5)
    P_elt = params.get('P_elt', 0.01)
    RH_sat = params.get('RH_sat', 0.6)
    Cv_sat = 0.03

    # --- Biological parameters ---
    Pmax = params.get('Pmax', 0.01)
    k_CO2 = params.get('k_CO2', 0.05)
    T_opt = params.get('T_opt', 298.0)
    T_range = params.get('T_range', 10.0)
    Y_B = params.get('Y_B', 0.6)
    R_maint = params.get('R_maint', 0.002)

    transpiration_coeff = params.get('transpiration_coeff', 0.0005)
    uptake_coeff = params.get('uptake_coeff', 0.02)
    evap_coeff = params.get('evap_coeff', 0.001)

    # --- Crew (astronaut) parameters ---
    O2_cons = params.get('O2_cons', 0.84/86400 * 0.001)
    CO2_prod = params.get('CO2_prod', 1.0/86400 * 0.001)

    # --- Heat sources ---
    alpha_heat = params.get('alpha_heat', 0.5)
    Q_rad = alpha_heat * Pmax * f_photo  # Radiative heating in the photosynthetic zone
    eta_heat = params.get('eta_heat', 0.05)

    # --- Wood Wide Web parameters ---
    gamma = params.get('gamma', 0.1)
    n_hill = params.get('n_hill', 2)
    Km = params.get('Km', 0.01)
    k_basal = params.get('k_basal', 1e-5)

    lambda_S = params.get('lambda_S', 0.1)
    k_prod = params.get('k_prod', 0.01)
    P_seuil = params.get('P_seuil', 1e-6)

    gamma_M = params.get('gamma_M', 0.05)
    delta_M = params.get('delta_M', 0.01)

    def equations(vars):
        theta, Cv, T, O2, CO2, B, S, M = vars

        # --- Quadratic penalties to stay within physical domain ---
        penalty = 0.0

        if theta < 0.01:    penalty += 1000.0 * (theta - 0.01)**2
        elif theta > 0.99:  penalty += 1000.0 * (theta - 0.99)**2
        if Cv < 0.0:        penalty += 1000.0 * Cv**2
        elif Cv > 0.05:     penalty += 1000.0 * (Cv - 0.05)**2
        if T < 280.0:       penalty += 10.0 * (T - 280.0)**2
        elif T > 320.0:     penalty += 10.0 * (T - 320.0)**2
        if O2 < 0.0:        penalty += 1000.0 * O2**2
        if CO2 < 0.0:       penalty += 1000.0 * CO2**2
        if B < 0.0:         penalty += 10000.0 * B**2
        if S < 0.0:         penalty += 10000.0 * S**2
        if M < 0.0:         penalty += 10000.0 * M**2

        # --- Physics (using volume fractions) ---
        E = evap_coeff * theta * (T - 273.0)
        U = uptake_coeff * theta * f_uptake
        RH = Cv / Cv_sat
        C_cond = k_cond * max(0.0, RH - RH_sat) * f_ceil
        T_r = transpiration_coeff * B * f_transp

        # --- Biology (using volume fractions) ---
        f_T = max(0.0, np.exp(-((T - T_opt) / T_range)**2))
        P_photo = Pmax * (CO2 / (CO2 + k_CO2)) * f_T * theta * B * f_photo

        resp_factor = max(0.1, 1.0 + 0.05 * (T - T_opt))
        R = R_maint * B * resp_factor * f_resp

        # --- Heat ---
        Q_bio = eta_heat * R
        Q_rad_absorbed = Q_rad

        # --- Wood Wide Web ---
        stress = max(0.0, P_seuil - P_photo)
        prod_S = k_prod * stress + k_basal  # Basal + stress-induced production

        # Cooperative term (Hill function)
        S_safe = max(S, 0.0)
        denom = Km**n_hill + S_safe**n_hill
        hill_term = S_safe**n_hill / denom if denom > 0 else 0.0
        cooperation = gamma * hill_term * B

        # --- Balance equations (8 equations) ---
        eq1 = -U - E + C_cond + penalty           # dθ/dt = 0
        eq2 = E + T_r - C_cond                       # dCv/dt = 0
        eq3 = Q_bio + Q_rad_absorbed - Lv * C_cond - P_elt * f_ceil  # dT/dt = 0
        eq4 = P_photo - R - O2_cons                  # dO2/dt = 0
        eq5 = -P_photo + R + CO2_prod                # dCO2/dt = 0
        eq6 = Y_B * P_photo - R + cooperation        # dB/dt = 0
        eq7 = prod_S - lambda_S * S                  # dS/dt = 0
        eq8 = gamma_M * S - delta_M * M              # dM/dt = 0

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]

    # --- Multiple initial guesses to avoid local minima ---
    # If a forced guess is provided, use it exclusively.
    force_guess = params.get('force_guess', None)
    if force_guess is not None:
        if len(force_guess) == 8:
            guesses = [list(force_guess)]
        else:
            print("Warning: force_guess must have 8 elements. Using default guesses.")
            guesses = [
                [0.3, 0.02, 298.0, 0.21, 0.001, 0.1, 0.0, 0.0],
                [0.3, 0.02, 298.0, 0.21, 0.001, 0.1, 0.01, 0.1],
                [0.5, 0.03, 295.0, 0.20, 0.002, 0.2, 0.001, 0.01],
                [0.2, 0.01, 305.0, 0.22, 0.0005, 0.05, 0.1, 0.5],
                [0.4, 0.025, 300.0, 0.19, 0.003, 0.15, 0.005, 0.05],
            ]
    else:
        # Default list (with and without mycorrhizal impulse)
        guesses = [
            [0.3, 0.02, 298.0, 0.21, 0.001, 0.1, 0.0, 0.0],       # Without WWW
            [0.3, 0.02, 298.0, 0.21, 0.001, 0.1, 0.01, 0.1],      # With impulse
            [0.5, 0.03, 295.0, 0.20, 0.002, 0.2, 0.001, 0.01],
            [0.2, 0.01, 305.0, 0.22, 0.0005, 0.05, 0.1, 0.5],
            [0.4, 0.025, 300.0, 0.19, 0.003, 0.15, 0.005, 0.05],
        ]

    best_solution = None
    best_residual = np.inf

    for guess in guesses:
        try:
            sol = fsolve(equations, guess, maxfev=20000, xtol=1e-12, full_output=True)
            solution = sol[0]
            residual = np.sum(np.array(equations(solution))**2)

            if (0.005 <= solution[0] <= 0.99 and
                0.0 <= solution[1] <= 0.05 and
                280.0 <= solution[2] <= 320.0 and
                0.0 <= solution[3] <= 1.0 and
                0.0 <= solution[4] <= 1.0 and
                solution[5] >= 0.0 and
                solution[6] >= 0.0 and
                solution[7] >= 0.0 and
                residual < best_residual):
                best_solution = solution
                best_residual = residual
        except Exception:
            continue

    if best_solution is None:
        return None

    return np.array(best_solution)


# ===============================
# 3. JACOBIAN MATRIX (8×8) — Consistent with volume fractions
# ===============================

def compute_jacobian_www(ss, params, eps=1e-8):
    """8×8 Jacobian for the system with WWW and volume fractions."""
    n = len(ss)
    J = np.zeros((n, n))

    # --- Volume fractions ---
    vol_frac = compute_volume_fractions(params)
    f_ceil   = vol_frac['f_ceil']
    f_photo  = vol_frac['f_photo']
    f_resp   = vol_frac['f_resp']
    f_transp = vol_frac['f_transp']
    f_uptake = vol_frac['f_uptake']

    # --- Physical parameters ---
    Lv = params.get('Lv', 0.5)
    k_cond = params.get('k_cond', 0.5)
    P_elt = params.get('P_elt', 0.01)
    RH_sat = params.get('RH_sat', 0.6)
    Cv_sat = 0.03

    # --- Biological parameters ---
    Pmax = params.get('Pmax', 0.01)
    k_CO2 = params.get('k_CO2', 0.05)
    T_opt = params.get('T_opt', 298.0)
    T_range = params.get('T_range', 10.0)
    Y_B = params.get('Y_B', 0.6)
    R_maint = params.get('R_maint', 0.002)

    transpiration_coeff = params.get('transpiration_coeff', 0.0005)
    uptake_coeff = params.get('uptake_coeff', 0.02)
    evap_coeff = params.get('evap_coeff', 0.001)

    O2_cons = params.get('O2_cons', 0.84/86400 * 0.001)
    CO2_prod = params.get('CO2_prod', 1.0/86400 * 0.001)

    alpha_heat = params.get('alpha_heat', 0.5)
    Q_rad = alpha_heat * Pmax * f_photo
    eta_heat = params.get('eta_heat', 0.05)

    # --- Wood Wide Web ---
    gamma = params.get('gamma', 0.1)
    n_hill = params.get('n_hill', 2)
    Km = params.get('Km', 0.01)
    k_basal = params.get('k_basal', 1e-5)

    lambda_S = params.get('lambda_S', 0.1)
    k_prod = params.get('k_prod', 0.01)
    P_seuil = params.get('P_seuil', 1e-6)

    gamma_M = params.get('gamma_M', 0.05)
    delta_M = params.get('delta_M', 0.01)

    def f(vars):
        theta, Cv, T, O2, CO2, B, S, M = vars

        # --- Physics (consistent with homogeneous_steady_state_www) ---
        E = evap_coeff * theta * (T - 273.0)
        U = uptake_coeff * theta * f_uptake
        RH = Cv / max(Cv_sat, 1e-10)
        C_cond = k_cond * max(0.0, RH - RH_sat) * f_ceil
        T_r = transpiration_coeff * max(0.0, B) * f_transp

        # --- Biology ---
        f_T = max(0.0, np.exp(-((T - T_opt) / T_range)**2))
        P_photo = Pmax * (max(0.0, CO2) / (max(0.0, CO2) + k_CO2)) * f_T * max(0.0, theta) * max(0.0, B) * f_photo

        resp_factor = max(0.1, 1.0 + 0.05 * (T - T_opt))
        R = R_maint * max(0.0, B) * resp_factor * f_resp

        Q_bio = eta_heat * R

        # --- Wood Wide Web ---
        stress = max(0.0, P_seuil - P_photo)
        prod_S = k_prod * stress + k_basal

        S_safe = max(S, 0.0)
        denom = Km**n_hill + S_safe**n_hill
        hill_term = S_safe**n_hill / denom if denom > 0 else 0.0
        cooperation = gamma * hill_term * max(0.0, B)

        # --- Time derivatives ---
        dtheta_dt = -U - E + C_cond
        dCv_dt = E + T_r - C_cond
        dT_dt = Q_bio + Q_rad - Lv * C_cond - P_elt * f_ceil
        dO2_dt = P_photo - R - O2_cons
        dCO2_dt = -P_photo + R + CO2_prod
        dB_dt = Y_B * P_photo - R + cooperation
        dS_dt = prod_S - lambda_S * S
        dM_dt = gamma_M * S - delta_M * M

        return np.array([dtheta_dt, dCv_dt, dT_dt, dO2_dt, dCO2_dt, dB_dt, dS_dt, dM_dt])

    f0 = f(ss)

    for j in range(n):
        ss_pert = ss.copy()
        ss_pert[j] += eps
        f_pert = f(ss_pert)
        J[:, j] = (f_pert - f0) / eps

    return J


# ===============================
# 4. MAIN EXECUTION
# ===============================
if __name__ == "__main__":

    params = {
        # --- Geometry ---
        'x_split': 0.5,
        'z_ceil': 0.9,
        'z_sol_continental': 0.4,
        'z_plant_max_continental': 0.8,
        'z_sol_marsh': 0.2,
        'z_plant_max_marsh': 0.6,

        # --- Physics ---
        'P_elt': 0.01,
        'k_cond': 0.5,
        'Lv': 0.5,
        'RH_sat': 0.6,

        # --- Biology ---
        'Pmax': 0.01,
        'k_CO2': 0.05,
        'T_opt': 298.0,
        'T_range': 10.0,
        'Y_B': 0.6,
        'R_maint': 0.002,

        # --- Transport ---
        'transpiration_coeff': 0.0005,
        'uptake_coeff': 0.02,
        'evap_coeff': 0.001,

        # --- Crew ---
        'O2_cons': 0.84/86400 * 0.001,
        'CO2_prod': 1.0/86400 * 0.001,

        # --- Heat ---
        'alpha_heat': 0.5,
        'eta_heat': 0.05,

        # --- Wood Wide Web ---
        'gamma': 0.1,
        'n_hill': 2,
        'Km': 0.01,
        'lambda_S': 0.1,
        'k_prod': 0.01,
        'P_seuil': 1e-6,
        'gamma_M': 0.05,
        'delta_M': 0.01,
        'k_basal': 1e-5,
    }

    # --- Display volume fractions ---
    vol_frac = compute_volume_fractions(params)
    print("=" * 60)
    print("VOLUME FRACTIONS")
    print("=" * 60)
    for key, val in vol_frac.items():
        print(f"  {key:12s} = {val:.4f}")

    # --- Steady state ---
    ss = homogeneous_steady_state_www(params)

    if ss is None:
        print("\nNo physical steady state found with these parameters.")
    else:
        var_names = ['θ', 'Cv', 'T', 'O2', 'CO2', 'B', 'S', 'M']
        print("\n" + "=" * 60)
        print("HOMOGENEOUS STEADY STATE (with Wood Wide Web)")
        print("=" * 60)
        for name, val in zip(var_names, ss):
            print(f"  {name:5s} = {val:.6f}")

        J = compute_jacobian_www(ss, params)
        eigenvalues, eigenvectors = np.linalg.eig(J)

        print("\n" + "=" * 60)
        print("EIGENVALUES")
        print("=" * 60)
        for i, ev in enumerate(eigenvalues):
            if ev.real > 1e-10:
                stability = "UNSTABLE  ←"
            elif ev.real < -1e-10:
                stability = "STABLE"
            else:
                stability = "MARGINAL"
            print(f"  λ{i+1} = {ev.real:+.6f} {ev.imag:+.6f}i  → {stability}")

        max_re = max(ev.real for ev in eigenvalues)
        print(f"\nMaximum real part: {max_re:.6f}")

        # Dominant unstable direction
        idx_max = np.argmax([ev.real for ev in eigenvalues])
        print(f"\nMost unstable mode (λ{idx_max+1}):")
        vec = eigenvectors[:, idx_max]
        for j, name in enumerate(var_names):
            if abs(vec[j]) > 0.01:
                print(f"    {name}: {vec[j].real:+.4f}")

    # --- Gamma sweep ---
    print("\n" + "=" * 60)
    print("γ SWEEP (Cooperation coefficient)")
    print("=" * 60)

    gamma_values = np.linspace(-0.05, 0.3, 35)
    max_re_parts = []

    for gamma in gamma_values:
        params['gamma'] = gamma
        # Force an initial impulse to explore the "On" state
        params['force_guess'] = [0.3, 0.02, 298.0, 0.21, 0.001, 0.1, 0.01, 0.1]
        ss_tmp = homogeneous_steady_state_www(params)
        if ss_tmp is not None:
            J_tmp = compute_jacobian_www(ss_tmp, params)
            evs = np.linalg.eigvals(J_tmp)
            max_re_parts.append(max(ev.real for ev in evs))
            # Also display S and M
            print(f"  γ = {gamma:+.3f}  |  max(Re(λ)) = {max_re_parts[-1]:+.6f}  |  S = {ss_tmp[6]:.6f}, M = {ss_tmp[7]:.6f}")
        else:
            max_re_parts.append(np.nan)
            print(f"  γ = {gamma:+.3f}  |  NO SOLUTION")

    # Plot (English version)
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, max_re_parts, 'o-', linewidth=2, markersize=6, color='darkblue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Stability threshold')
    plt.axvline(x=0.0, color='green', linestyle='--', linewidth=1.5, label='Cooperation threshold')
    plt.xlabel('γ (Mycorrhizal cooperation coefficient)\n  (Competition <-- --> Cooperation)', fontsize=13)
    plt.ylabel('max(Re(λ))', fontsize=13)
    plt.title('Bifurcation diagram: Stability as a function of γ\nTECP system with Wood Wide Web', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure_Bifurcation_Gamma_V8.3_en.png', dpi=300)
    plt.savefig('Figure_Bifurcation_Gamma_V8.3_en.svg', format='svg')
    plt.show()

    print("\nFigure saved: Figure_Bifurcation_Gamma_V8.3_en.png")
    print("=" * 60)
    print("ANALYSIS COMPLETE.")
    print("=" * 60)