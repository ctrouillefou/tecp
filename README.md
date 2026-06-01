# tecp
Toroidal Ecological Coupled Platform
# TECP — Toroidal Ecological Coupled Platform

Supplementary code for the paper:

**"From Control to Cooperation: A Mathematical Proof That Fungal Networks Can Replace Reinforcement Learning in Closed Life Support Systems"**  
*Christophe Marcel Trouillefou*  
Submitted to *Acta Astronautica* (2026)

## Overview

The TECP is a 3D reaction‑diffusion‑chemotaxis model of a closed toroidal ecological reactor for bioregenerative life support. It couples plant physiology, water cycling, and a mechanistic mycorrhizal network (the "Wood Wide Web") to study whether fungal cooperation can replace external control.

This repository contains all the scripts necessary to reproduce the results presented in the paper.

## Files

| File | Description |
|:---|:---|
| `algo_2_3d_bio_V8.3.py` | Homogeneous steady‑state analysis: bifurcation sweep over $\gamma$, Jacobian eigenvalues, and dynamic On/Off simulation |
| `algo_3_3d_bio_V5_stochastic.py` | Full 3D stochastic simulation with Gaussian noise on Peltier power and day/night cycle |
| `algo_3_3d_bio_V5_deterministic.py` | Full 3D deterministic simulation over 5000 time steps (Figures 4–5 of the paper) |
| `build_masks.py` | Reusable Boolean mask builder for the five process‑specific spatial domains (imported by the 3D scripts) |
| `compute_volume_fractions.py` | Computes the effective volume fractions from geometric thresholds (used by the homogeneous analysis) |
| `plot_masks.py` | Visualisation script for the five Boolean masks (outputs `masks_TECP.png` and `.svg`) |

## Dependencies

- Python 3.10+
- NumPy 1.24+
- SciPy 1.11+
- Matplotlib 3.7+
- [FiPy](https://www.ctcms.nist.gov/fipy/) 3.4+ (for 3D simulations)

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/ctrouillefou/tecp.git
   cd tecp
   
2. Install the required packages:
   ```bash
   pip install numpy scipy matplotlib fipy

3. Run the homogeneous analysis:
   ```bash
   python algo_2_3d_bio_V8.3.py

4. Run the 3D deterministic simulation:
   ```bash
   python algo_3_3d_bio_V5_deterministic.py

5. Run the 3D stochastic simulation:
   ```bash
   python algo_3_3d_bio_V5_stochastic.py

License

This project is licensed under the GNU General Public License v3.0 — see the LICENSE file for details.
Citation

If you use this code in your research, please cite:

    Trouillefou, C.M. (2026). From Control to Cooperation: A Mathematical Proof That Fungal Networks Can Replace Reinforcement Learning in Closed Life Support Systems. Acta Astronautica, submitted.

The code is also archived on Zenodo with DOI: [to be inserted after publication].
Contact

Christophe Marcel Trouillefou — christophe.trouillefou@gmail.com
