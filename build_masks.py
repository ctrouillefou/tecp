# ===============================
# build_masks.py — Reusable Boolean mask builder for the TECP
# ===============================
import numpy as np
from fipy import CellVariable

def build_masks(mesh, params):
    """
    Construct the five Boolean masks from geometric parameters.

    Parameters
    ----------
    mesh : fipy.meshes.grid3D.Grid3D
        The 3D mesh on which masks are defined.
    params : dict
        Dictionary containing geometric thresholds:
        - 'x_split' : horizontal position separating continental (left) from marsh (right)
        - 'z_ceil'  : vertical threshold for Peltier module (z > z_ceil*z_max)
        - 'z_sol_continental' : soil surface height for continental biome
        - 'z_plant_max_continental' : max foliage height for continental biome
        - 'z_sol_marsh' : soil surface height for marsh biome
        - 'z_plant_max_marsh' : max foliage height for marsh biome

    Returns
    -------
    M_ceil, M_photo, M_resp, M_transp, M_uptake : tuple of CellVariable
        The five Boolean masks (1 where process is active, 0 elsewhere).
    """
    x = mesh.cellCenters[0]
    z = mesh.cellCenters[2]
    z_max = z.max()
    x_mid = params['x_split'] * x.max()

    continental = (x < x_mid)
    marsh = (x >= x_mid)

    # Continental thresholds
    z_sol_c = params['z_sol_continental'] * z_max
    z_max_c = params['z_plant_max_continental'] * z_max
    z_resp_c = (z_sol_c + z_max_c) / 2.0

    # Marsh thresholds
    z_sol_m = params['z_sol_marsh'] * z_max
    z_max_m = params['z_plant_max_marsh'] * z_max
    z_resp_m = (z_sol_m + z_max_m) / 2.0

    # 1. Peltier ceiling mask
    M_ceil = CellVariable(mesh=mesh, value=0.0)
    M_ceil.setValue(1.0, where=(z > params['z_ceil'] * z_max))

    # 2. Photosynthesis: lower half of foliage (z_sol ≤ z ≤ z_resp)
    M_photo = CellVariable(mesh=mesh, value=0.0)
    M_photo.setValue(1.0, where=(continental & (z >= z_sol_c) & (z <= z_resp_c)) |
                                (marsh       & (z >= z_sol_m) & (z <= z_resp_m)))

    # 3. Respiration: foliage + roots (up to mid-height)
    M_resp = CellVariable(mesh=mesh, value=0.0)
    M_resp.setValue(1.0, where=(continental & (z <= z_resp_c)) |
                                (marsh       & (z <= z_resp_m)))

    # 4. Transpiration: lower canopy (same vertical range as M_photo)
    M_transp = CellVariable(mesh=mesh, value=0.0)
    M_transp.setValue(1.0, where=(continental & (z >= z_sol_c) & (z <= z_resp_c)) |
                                  (marsh       & (z >= z_sol_m) & (z <= z_resp_m)))

    # 5. Root uptake: below soil surface
    M_uptake = CellVariable(mesh=mesh, value=0.0)
    M_uptake.setValue(1.0, where=(continental & (z <= z_sol_c)) |
                                  (marsh       & (z <= z_sol_m)))

    return M_ceil, M_photo, M_resp, M_transp, M_uptake