#!/usr/bin/env python3
"""Regenera la Figura B.1 (cantidad de días sin dato por estación) con el
mismo estilo "revista científica" del resto de las figuras de la tesis
(ver scripts/generar_figuras_resultados.py). No existía un script fuente
para esta figura en el repositorio; los valores provienen de la Tabla 3.1
(03_metodologia.tex)."""
import matplotlib.pyplot as plt

OUT = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_GRID = "#d8d8d4"
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#52514e"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "legend.fontsize": 8, "axes.edgecolor": COLOR_MUTED, "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT, "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED,
    "axes.grid": True, "grid.color": COLOR_GRID, "grid.linewidth": 0.6,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Estación, dias sin dato, % (Tabla 3.1), ordenadas de mayor a menor
datos = [
    ("Asunción", 779, 8.20),
    ("San Juan", 277, 2.92),
    ("Jesús", 234, 2.46),
    ("A. Weberbauer", 138, 1.45),
    ("Namora", 107, 1.13),
    ("La Encañada", 62, 0.65),
]
nombres = [d[0] for d in datos]
dias = [d[1] for d in datos]
pct = [d[2] for d in datos]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
y = range(len(nombres))
ax.barh(y, dias, color=COLOR_LSTM, edgecolor="white", linewidth=0.4, height=0.62, zorder=3)
ax.set_yticks(list(y))
ax.set_yticklabels(nombres)
ax.invert_yaxis()  # la barra más larga arriba, igual que la Tabla 3.1
ax.set_xlabel("Días sin dato")
ax.set_xlim(0, max(dias) * 1.30)
ax.set_title("Cantidad de días sin dato por estación (2000–2025)",
             loc="left", fontweight="bold")
ax.grid(axis="x", zorder=0)
ax.grid(axis="y", visible=False)

for yi, d, p in zip(y, dias, pct):
    ax.text(d + max(dias) * 0.02, yi, f"{d} días ({p:.2f} %)",
            va="center", ha="left", fontsize=8, color=COLOR_TEXT)

fig.tight_layout()
fig.savefig(f"{OUT}/anexo_datos_faltantes.pdf")
fig.savefig(f"{OUT}/anexo_datos_faltantes.png")
plt.close(fig)
print("-> anexo_datos_faltantes.pdf / .png")
