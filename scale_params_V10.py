#!/usr/bin/env python3
# =====================================================================
# SCALE_PARAMS_V10 — RE‑DIMENSIONNALISATION FROM DIMENSIONLESS TO REAL UNITS
# 
# Description:
#   Converts dimensionless model parameters (params_adim.json) into
#   real‑world units (params_real.json) using experimentally informed
#   scale factors. Applies different time scales for fast (hours) and
#   slow (days) processes.
#
# Version: V10.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. Uses tau_base = 3600 s as the baseline.
#   2. Scale factors for photosynthesis, respiration, mycelial growth, etc.
#   3. Preserves dimensionless parameters (γ, Km, n_hill, etc.).
#   4. Generates params_real.json for use in V10 simulations.
# =====================================================================

import json
import argparse
import sys

# --- Scale factors s_i = tau_i / tau_base, tau_base = 3600 s ---
SCALE_FACTORS = {
    "photosynthesis": 1.0,       # 1 hour
    "respiration": 1.0,          # 1 hour
    "evapotranspiration": 1.0,   # 1 hour
    "condensation": 1.0,         # 1 hour
    "thermal": 1.0,              # 1 hour
    "mycelial_growth": 168.0,    # 168 hours (1 week)
    "mycorrhizal_signal": 168.0, # 168 hours (1 week)
    "biomass_growth": 168.0,     # 168 hours (1 week)
    "soil_evolution": 720.0,     # 720 hours (1 month)
}

# --- Mapping between your parameter names and the families ---
PARAM_MAP = {
    # Photosynthesis
    "Pmax": "photosynthesis",
    # Respiration
    "R_maint": "respiration",
    # Evapotranspiration (transpiration + evaporation + uptake)
    "transpiration_coeff": "evapotranspiration",
    "evap_coeff": "evapotranspiration",
    "uptake_coeff": "evapotranspiration",
    # Condensation / Peltier
    "k_cond": "condensation",
    "P_elt": "thermal",
    "Lv": "thermal",
    # Mycorrhizae (growth and signal)
    "gamma_M": "mycelial_growth",
    "delta_M": "mycelial_growth",
    "lambda_S": "mycorrhizal_signal",
    "k_prod": "mycorrhizal_signal",
    "k_basal": "mycorrhizal_signal",
    # Biomass
    "Y_B": "biomass_growth",
    # Soil
    "delta_soil": "soil_evolution",
}

# Parameters that should not be modified (already in SI units, or dimensionless)
KEEP_UNCHANGED = [
    "gamma",          # cooperation (dimensionless)
    "Km",             # Hill constant (dimensionless)
    "n_hill",         # Hill exponent (dimensionless)
    "P_seuil",        # stress threshold (dimensionless)
    "O2_cons",        # already in s^-1
    "CO2_prod",       # already in s^-1
    "alpha_heat",     # dimensionless
    "eta_heat",       # dimensionless
    "x_split",        # geometric (dimensionless)
    "z_ceil",         # geometric
    "z_sol_continental", "z_plant_max_continental",
    "z_sol_marsh", "z_plant_max_marsh",
    "RH_sat",         # relative humidity at saturation (dimensionless)
    "T_opt", "T_range",  # temperatures (K) — left as is
]

def get_scale_factor(param_name):
    """Return the scale factor for a given parameter, or 1.0 by default."""
    family = PARAM_MAP.get(param_name, None)
    if family is None:
        # If the parameter is not in the map, assume it is dimensionless
        # and apply a factor of 1.0 (i.e., no change)
        return 1.0
    return SCALE_FACTORS.get(family, 1.0)

def scale_parameter(param_name, value, tau_base=3600.0):
    """
    Apply the scale factor.
    For a rate (s^-1): k_real = k_adim / (tau_base * s_i)
    For a time constant (s): tau_real = tau_adim * (tau_base * s_i)
    """
    # If the parameter is in the unchanged list, return the original value
    if param_name in KEEP_UNCHANGED:
        return value

    s_i = get_scale_factor(param_name)

    # Automatic detection: if the name contains 'tau' or 'time', it is a time constant
    if 'tau' in param_name.lower() or 'time' in param_name.lower():
        return value * tau_base * s_i
    else:
        # Otherwise, treat it as a rate (1/s)
        return value / (tau_base * s_i)

def main():
    parser = argparse.ArgumentParser(
        description='Re‑dimensionnalisation of the TECP model (V10.0)'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='JSON file with dimensionless parameters')
    parser.add_argument('--output', '-o', required=True,
                        help='JSON file with re‑dimensionnalised parameters')
    parser.add_argument('--tau_base', '-t', type=float, default=3600.0,
                        help='Base time scale in seconds (default: 3600 s)')
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            params_adim = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{args.input}' is not a valid JSON.")
        sys.exit(1)

    params_real = {}
    for key, value in params_adim.items():
        if isinstance(value, (int, float)):
            params_real[key] = scale_parameter(key, value, args.tau_base)
        else:
            # Pass through nested structures (lists, dicts) unchanged
            params_real[key] = value

    # Metadata
    params_real['_metadata'] = {
        'tau_base': args.tau_base,
        'scale_factors': SCALE_FACTORS,
        'description': 'Re‑dimensionnalised parameters for the TECP model V10.0',
        'formula': "For a rate: k_real = k_adim / (tau_base * s_i)\n"
                   "For a time constant: tau_real = tau_adim * (tau_base * s_i)"
    }

    with open(args.output, 'w') as f:
        json.dump(params_real, f, indent=2)

    print(f"✅ File generated: {args.output}")
    print(f"   tau_base = {args.tau_base} s")
    print("   Scale factors applied:")
    for fam, s in SCALE_FACTORS.items():
        print(f"     - {fam}: {s}")
    print("   Parameters kept unchanged:")
    for p in KEEP_UNCHANGED:
        if p in params_adim:
            print(f"     - {p}")

if __name__ == "__main__":
    main()