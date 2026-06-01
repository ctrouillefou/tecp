# ===============================
# plot_masks.py — Visualise the five Boolean masks of the TECP
# ===============================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

biomes = ['Continental (left)', 'Marsh (right)']
z_sol = [0.4, 0.2]          # soil surface (normalised)
z_max = [0.8, 0.6]          # maximum foliage height
z_resp = [0.6, 0.4]         # mid‑height of the foliage (z_sol + z_max)/2

for i, biome in enumerate(biomes):
    ax[i].set_xlim(0, 1)
    ax[i].set_ylim(0, 1)
    ax[i].set_title(biome)
    ax[i].set_xticks([])
    ax[i].set_ylabel('z (normalised)')

    # Soil surface line
    ax[i].axhline(y=z_sol[i], color='brown', linestyle='--', linewidth=2, label='Soil surface')

    # M_ceil: Peltier ceiling (z > 0.9)
    ax[i].axhspan(0.9, 1.0, color='lightblue', alpha=0.3)
    ax[i].text(0.5, 0.95, "M_ceil = 1", ha='center', va='center', fontsize=8)

    # M_photo: lower foliage (z_sol to z_resp) -- CORRECTED
    ax[i].axhspan(z_sol[i], z_resp[i], color='green', alpha=0.15)
    ax[i].text(0.5, (z_sol[i] + z_resp[i]) / 2, "M_photo = 1", ha='center', va='center', fontsize=8)

    # M_resp: up to mid‑height (0 to z_resp)
    ax[i].axhspan(0, z_resp[i], color='orange', alpha=0.1)
    ax[i].text(0.5, z_resp[i] / 2, "M_resp = 1", ha='center', va='center', fontsize=8)

    # M_transp: lower canopy (z_sol to z_resp)
    ax[i].axhspan(z_sol[i], z_resp[i], color='purple', alpha=0.1)
    ax[i].text(0.5, (z_sol[i] + z_resp[i]) / 2, "M_transp = 1", ha='center', va='center', fontsize=8)

    # M_uptake: root zone (0 to z_sol)
    ax[i].axhspan(0, z_sol[i], color='brown', alpha=0.2)
    ax[i].text(0.5, z_sol[i] / 2, "M_uptake = 1", ha='center', va='center', fontsize=8)

    ax[i].legend(loc='lower right')

plt.tight_layout()
plt.savefig('masks_TECP.png', dpi=300)
plt.savefig('masks_TECP.svg', format='svg')
plt.show()