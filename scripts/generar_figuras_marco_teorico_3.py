#!/usr/bin/env python3
"""
Figuras conceptuales para la ampliación del Marco Teórico (Bases teóricas):
balance hídrico de cuenca, comparación bagging vs. boosting, distribución de
Gumbel para análisis de frecuencia de extremos, y función de pérdida asimétrica
frente al MSE simétrico. Mismo estilo que generar_figuras_marco_teorico.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_GREEN = "#1baf7a"
COLOR_OBS = "#1a1a1a"
COLOR_GRID = "#d8d8d4"
COLOR_MUTED = "#52514e"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": COLOR_MUTED, "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a", "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED,
    "axes.grid": True, "grid.color": COLOR_GRID, "grid.linewidth": 0.6,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    fig.savefig(f"{OUT}/{nombre}.png")
    plt.close(fig)
    print(f"  -> {nombre}.pdf / .png")


# ---------------------------------------------------------------------------
# 1. Balance hídrico esquemático de una cuenca
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Cuenca (trapecio de terreno)
terreno = mpatches.Polygon(
    [(0.6, 0.6), (9.4, 0.6), (7.8, 2.6), (2.2, 2.6)],
    closed=True, facecolor="#efe9d8", edgecolor=COLOR_MUTED, linewidth=1.2, zorder=1,
)
ax.add_patch(terreno)
# Acuifero (almacenamiento subterraneo)
acuifero = mpatches.FancyBboxPatch(
    (2.6, 0.75), 4.8, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor="#cfe3f5", edgecolor=COLOR_LSTM, linewidth=1.0, zorder=2,
)
ax.add_patch(acuifero)
ax.text(5.0, 1.02, r"Almacenamiento ($\Delta S$)", ha="center", va="center", fontsize=7.8, color=COLOR_LSTM)

# Nube (compuesta por circulos superpuestos) y flecha de precipitacion (P)
for dx, dy, r in [(-0.35, 0, 0.32), (0.05, 0.15, 0.4), (0.45, 0, 0.3), (0.1, -0.12, 0.34)]:
    ax.add_patch(mpatches.Circle((2.6 + dx, 5.15 + dy), r, facecolor="#d8d8d4",
                                  edgecolor=COLOR_MUTED, linewidth=0.8, zorder=2))
ax.add_patch(FancyArrowPatch((2.6, 4.75), (3.1, 2.7), arrowstyle="-|>", mutation_scale=14,
                              linewidth=1.6, color=COLOR_LSTM))
ax.text(1.9, 3.7, "Precipitación\n$(P)$", ha="center", fontsize=8.2, color=COLOR_LSTM, fontweight="bold")

# Evapotranspiracion (ET) hacia arriba
ax.add_patch(FancyArrowPatch((5.6, 2.55), (5.9, 4.6), arrowstyle="-|>", mutation_scale=14,
                              linewidth=1.6, color=COLOR_XGB, connectionstyle="arc3,rad=-0.25"))
ax.text(7.3, 4.55, "Evapotranspiración\n$(ET)$", ha="center", fontsize=8.2, color=COLOR_XGB, fontweight="bold")

# Escorrentia superficial (Q) saliendo lateralmente
ax.add_patch(FancyArrowPatch((7.9, 1.6), (9.6, 1.0), arrowstyle="-|>", mutation_scale=14,
                              linewidth=1.6, color=COLOR_GREEN))
ax.text(9.05, 2.05, "Escorrentía\n$(Q)$", ha="center", fontsize=8.2, color=COLOR_GREEN, fontweight="bold")

# Infiltracion (flecha corta hacia el acuifero)
ax.add_patch(FancyArrowPatch((3.4, 2.2), (3.4, 1.35), arrowstyle="-|>", mutation_scale=11,
                              linewidth=1.3, color=COLOR_MUTED, linestyle=(0, (3, 2))))
ax.text(3.9, 1.75, "Infiltración", ha="left", fontsize=7.3, color=COLOR_MUTED, style="italic")

ax.text(5.0, 0.15, r"Ecuación de balance hídrico:  $P = ET + Q + \Delta S$",
        ha="center", fontsize=9.3, fontweight="bold", color="#1a1a1a")

guardar(fig, "fig_balance_hidrico")

# ---------------------------------------------------------------------------
# 2. Bagging vs. Boosting (esquema comparativo)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

rng = np.random.default_rng(3)

# --- Bagging: arboles independientes en paralelo, voto/promedio ---
ax = axes[0]
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.text(5, 7.5, "Bagging (paralelo)", ha="center", fontsize=9.6, fontweight="bold")
ax.add_patch(FancyBboxPatch((3.5, 6.0), 3.0, 0.8, boxstyle="round,pad=0.05", facecolor="#efe9df",
                             edgecolor=COLOR_MUTED, linewidth=1.0))
ax.text(5.0, 6.4, "Datos (remuestreo\nbootstrap)", ha="center", va="center", fontsize=6.8)
ys = 4.3
xs = [1.6, 5.0, 8.4]
for i, x in enumerate(xs):
    ax.add_patch(FancyArrowPatch((5.0, 6.0), (x, ys + 0.8), arrowstyle="-|>", mutation_scale=9,
                                  linewidth=1.0, color=COLOR_MUTED))
    ax.add_patch(FancyBboxPatch((x - 0.9, ys), 1.8, 0.8, boxstyle="round,pad=0.02",
                                 facecolor="#cfe3f5", edgecolor=COLOR_LSTM, linewidth=1.1))
    ax.text(x, ys + 0.4, f"Árbol {i+1}\n(independiente)", ha="center", va="center", fontsize=6.6)
    ax.add_patch(FancyArrowPatch((x, ys), (5.0, 2.0), arrowstyle="-|>", mutation_scale=9,
                                  linewidth=1.0, color=COLOR_MUTED))
ax.add_patch(FancyBboxPatch((3.7, 1.2), 2.6, 0.8, boxstyle="round,pad=0.02", facecolor="#e2f5ec",
                             edgecolor=COLOR_GREEN, linewidth=1.2))
ax.text(5.0, 1.6, "Promedio / voto\nde mayoría", ha="center", va="center", fontsize=6.8, fontweight="bold")

# --- Boosting: arboles secuenciales que corrigen el residuo anterior ---
ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.text(5, 7.5, "Boosting (secuencial)", ha="center", fontsize=9.6, fontweight="bold")
xs2 = [1.6, 5.0, 8.4]
etiquetas = ["Árbol 1\najusta $y$", "Árbol 2\najusta el residuo\nde (1)", "Árbol 3\najusta el residuo\nde (1)+(2)"]
for i, (x, et) in enumerate(zip(xs2, etiquetas)):
    ax.add_patch(FancyBboxPatch((x - 1.0, 4.3), 2.0, 1.4, boxstyle="round,pad=0.02",
                                 facecolor="#fbe4d6", edgecolor=COLOR_XGB, linewidth=1.1))
    ax.text(x, 5.0, et, ha="center", va="center", fontsize=6.6)
    if i < 2:
        ax.add_patch(FancyArrowPatch((x + 1.0, 5.0), (xs2[i+1] - 1.0, 5.0), arrowstyle="-|>",
                                      mutation_scale=11, linewidth=1.3, color=COLOR_MUTED))
        ax.text((x + xs2[i+1]) / 2, 5.55, "residuo", ha="center", fontsize=6.2,
                style="italic", color=COLOR_MUTED)
for x_arbol, x_suma in zip(xs2, [4.0, 5.0, 6.0]):
    ax.add_patch(FancyArrowPatch((x_arbol, 4.3), (x_suma, 2.0), arrowstyle="-|>", mutation_scale=11,
                                  linewidth=1.3, color=COLOR_MUTED))
ax.add_patch(FancyBboxPatch((3.5, 1.2), 3.0, 0.8, boxstyle="round,pad=0.02", facecolor="#e2f5ec",
                             edgecolor=COLOR_GREEN, linewidth=1.2))
ax.text(5.0, 1.6, "Suma ponderada\nde los árboles", ha="center", va="center", fontsize=6.8, fontweight="bold")

fig.tight_layout()
guardar(fig, "fig_bagging_boosting")

# ---------------------------------------------------------------------------
# 3. Distribucion de Gumbel para analisis de frecuencia de extremos
# ---------------------------------------------------------------------------
mu, beta = 40.0, 8.0
x = np.linspace(10, 100, 400)
z = (x - mu) / beta
pdf = (1 / beta) * np.exp(-(z + np.exp(-z)))
cdf = np.exp(-np.exp(-z))
T = 1 / (1 - cdf)

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
ax = axes[0]
ax.plot(x, pdf, color=COLOR_XGB, linewidth=1.8)
ax.fill_between(x, pdf, color=COLOR_XGB, alpha=0.12)
ax.set_xlabel("Precipitación máxima anual (mm, esquemático)")
ax.set_ylabel("Densidad de probabilidad")
ax.set_title("Función de densidad de Gumbel", loc="left", fontsize=9.3, fontweight="bold")

ax = axes[1]
mask = (T > 1.01) & (T < 500)
ax.plot(T[mask], x[mask], color=COLOR_LSTM, linewidth=1.8)
for Tr in [10, 50, 100]:
    idx = np.argmin(np.abs(T - Tr))
    ax.axvline(Tr, color=COLOR_GRID, linewidth=0.8, linestyle=":")
    ax.plot(T[idx], x[idx], "o", color=COLOR_MUTED, markersize=4, zorder=3)
    ax.annotate(f"T={Tr} años", (T[idx], x[idx]), textcoords="offset points",
                xytext=(6, -10), fontsize=6.8, color=COLOR_MUTED)
ax.set_xscale("log")
ax.set_xlabel("Periodo de retorno $T$ (años, escala log.)")
ax.set_ylabel("Magnitud estimada (mm)")
ax.set_title("Curva de frecuencia asociada", loc="left", fontsize=9.3, fontweight="bold")

fig.tight_layout()
guardar(fig, "fig_distribucion_gumbel")

# ---------------------------------------------------------------------------
# 4. Funcion de perdida asimetrica vs. MSE simetrico
# ---------------------------------------------------------------------------
residuo = np.linspace(-4, 4, 400)  # residuo = y_obs - y_pred
mse = residuo ** 2
peso = np.where(residuo > 0, 2.0, 1.0)  # subestimacion (residuo>0) penalizada 2x
asimetrica = peso * residuo ** 2

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(residuo, mse, color=COLOR_MUTED, linewidth=1.6, linestyle="--", label=r"MSE simétrico: $(y-\hat{y})^2$")
ax.plot(residuo, asimetrica, color=COLOR_XGB, linewidth=2.0,
        label=r"Pérdida asimétrica: $2\times(y-\hat{y})^2$ si subestima")
ax.axvline(0, color=COLOR_GRID, linewidth=0.8)
ax.axvspan(0, 4, color=COLOR_XGB, alpha=0.06)
ax.text(2.0, ax.get_ylim()[1] * 0.85, "Subestimación\n($y > \\hat{y}$)\npenalizada más",
        ha="center", fontsize=7.4, color=COLOR_XGB, style="italic")
ax.set_xlabel(r"Residuo $(y_{obs} - y_{pred})$")
ax.set_ylabel("Valor de la función de pérdida")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=7.6)
fig.tight_layout()
guardar(fig, "fig_perdida_asimetrica")

print("Figuras conceptuales de la ampliación del marco teórico generadas.")
