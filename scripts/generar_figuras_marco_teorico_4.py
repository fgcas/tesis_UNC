#!/usr/bin/env python3
"""
Figuras conceptuales para la segunda ampliación del Marco Teórico (Bases teóricas):
modelos conceptuales vs. basados en datos, interpolación espacial (IDW vs. Kriging),
equifinalidad/incertidumbre, fuga de datos, grid search vs. random search, y la
maldición de la dimensionalidad. Mismo estilo que los scripts anteriores.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_GREEN = "#1baf7a"
COLOR_GREEN_DARK = "#0c7a52"
COLOR_RED = "#c0392b"
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
# 1. Modelo conceptual (caja blanca) vs. modelo basado en datos (caja negra)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

# --- Caja blanca: compartimientos fisicos visibles ---
ax = axes[0]
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.text(5, 7.5, "Modelo conceptual (caja blanca)", ha="center", fontsize=11.5, fontweight="bold")
ax.add_patch(FancyArrowPatch((5.0, 7.1), (5.0, 6.3), arrowstyle="-|>", mutation_scale=11, color=COLOR_LSTM))
ax.text(5.6, 6.7, "$P$", color=COLOR_LSTM, fontsize=10.5)
ax.add_patch(FancyBboxPatch((2.2, 4.6), 5.6, 1.6, boxstyle="round,pad=0.02", facecolor="#cfe3f5",
                             edgecolor=COLOR_LSTM, linewidth=1.1))
ax.text(5.0, 5.4, "Almacenamiento de\nhumedad del suelo", ha="center", va="center", fontsize=9.5)
ax.add_patch(FancyArrowPatch((5.0, 4.55), (5.0, 3.35), arrowstyle="-|>", mutation_scale=11, color=COLOR_MUTED))
ax.add_patch(FancyBboxPatch((2.2, 2.0), 5.6, 1.35, boxstyle="round,pad=0.02", facecolor="#e2f5ec",
                             edgecolor=COLOR_GREEN, linewidth=1.1))
ax.text(5.0, 2.68, "Función de tránsito\n(enrutamiento, retardo)", ha="center", va="center", fontsize=9.5)
ax.add_patch(FancyArrowPatch((5.0, 1.95), (5.0, 0.9), arrowstyle="-|>", mutation_scale=11, color=COLOR_XGB))
ax.text(5.9, 1.2, "$PP$ estimado", color=COLOR_XGB, fontsize=10)
ax.text(5.0, 0.25, "Parámetros con significado físico\n(capacidad de campo, conductividad...)",
        ha="center", fontsize=8.5, style="italic", color=COLOR_MUTED)

# --- Caja negra: modelo basado en datos ---
ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.text(5, 7.5, "Modelo basado en datos (caja negra)", ha="center", fontsize=11.5, fontweight="bold")
entradas = ["Precip.\nobservada", "Precip.\nPISCO", "Forzamiento\nestacional"]
for i, (x, txt) in enumerate(zip([1.8, 5.0, 8.2], entradas)):
    ax.add_patch(FancyBboxPatch((x - 1.1, 5.7), 2.2, 1.0, boxstyle="round,pad=0.02", facecolor="#efe9df",
                                 edgecolor=COLOR_MUTED, linewidth=1.0))
    ax.text(x, 6.2, txt, ha="center", va="center", fontsize=8.8)
    ax.add_patch(FancyArrowPatch((x, 5.65), (5.0, 4.55), arrowstyle="-|>", mutation_scale=9, color=COLOR_MUTED))
ax.add_patch(FancyBboxPatch((2.0, 2.4), 6.0, 2.0, boxstyle="round,pad=0.03", facecolor="#2a2a28",
                             edgecolor=COLOR_OBS, linewidth=1.4))
ax.text(5.0, 3.4, "$f_\\theta(X)$", ha="center", va="center", fontsize=15, color="white", fontweight="bold")
ax.text(5.0, 2.75, "parámetros ajustados por optimización,\nsin interpretación física directa",
        ha="center", va="center", fontsize=8.3, color="#d8d8d4", style="italic")
ax.add_patch(FancyArrowPatch((5.0, 2.35), (5.0, 1.2), arrowstyle="-|>", mutation_scale=11, color=COLOR_XGB))
ax.text(5.9, 1.6, "$\\hat{y}$", color=COLOR_XGB, fontsize=10.5)
ax.text(5.0, 0.5, "LSTM y XGBoost de esta tesis pertenecen\na este segundo paradigma",
        ha="center", fontsize=8.8, style="italic", color=COLOR_MUTED)

fig.tight_layout()
guardar(fig, "fig_conceptual_vs_datos")

# ---------------------------------------------------------------------------
# 2. Interpolacion espacial: peso vs distancia, IDW vs Kriging
# ---------------------------------------------------------------------------
d = np.linspace(0.3, 10, 300)
peso_idw = 1 / d**1.3
peso_idw /= peso_idw.max()

# Kriging esquematico: peso derivado de un variograma esferico (no monotono en la practica
# por la covarianza cruzada entre estaciones, aqui simplificado para fines ilustrativos)
rango = 6.0
gamma = np.where(d < rango, 1.5 * (d / rango) - 0.5 * (d / rango) ** 3, 1.0)
peso_krig = (1 - gamma)
peso_krig = np.clip(peso_krig, 0, None)
peso_krig /= peso_krig.max()

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(d, peso_idw, color=COLOR_XGB, linewidth=1.8, label="IDW")
ax.plot(d, peso_krig, color=COLOR_LSTM, linewidth=1.8, label="Kriging")
ax.set_xlabel("Distancia a la estación vecina (km, esquemático)")
ax.set_ylabel("Peso relativo asignado a la estación vecina")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=7.6)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.05)
fig.tight_layout()
guardar(fig, "fig_interpolacion_idw_kriging")

# ---------------------------------------------------------------------------
# 3. Equifinalidad: multiples combinaciones de parametros con desempeno similar
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
n = 400
p1 = rng.uniform(0, 10, n)
p2 = rng.uniform(0, 10, n)
centro1, centro2 = (3.2, 6.5), (7.0, 3.0)
d1 = np.hypot(p1 - centro1[0], p2 - centro1[1])
d2 = np.hypot(p1 - centro2[0], p2 - centro2[1])
nse = 0.75 * np.exp(-(np.minimum(d1, d2) ** 2) / 6.0) + rng.normal(0, 0.02, n)
nse = np.clip(nse, -0.2, 0.8)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
sc = ax.scatter(p1, p2, c=nse, cmap="RdYlGn", s=22, edgecolors="none", vmin=-0.2, vmax=0.8)
umbral = nse > 0.55
ax.scatter(p1[umbral], p2[umbral], facecolors="none", edgecolors=COLOR_OBS, s=45, linewidths=0.9,
           label="Región equifinal (NSE > 0.55)")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("NSE simulado")
ax.set_xlabel("Valor del primer parámetro interno del modelo (esquemático)")
ax.set_ylabel("Valor del segundo parámetro interno del modelo (esquemático)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=7.4)
fig.tight_layout()
guardar(fig, "fig_equifinalidad")

# ---------------------------------------------------------------------------
# 4. Fuga de datos: flujo correcto vs. incorrecto de informacion train/test
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.0))

for ax, correcto in zip(axes, [False, True]):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.add_patch(FancyBboxPatch((0.3, 0.6), 5.4, 1.2, boxstyle="round,pad=0.02", facecolor="#cfe3f5",
                                 edgecolor=COLOR_LSTM, linewidth=1.1))
    ax.text(3.0, 1.2, "Entrenamiento", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyBboxPatch((6.0, 0.6), 3.7, 1.2, boxstyle="round,pad=0.02",
                                 facecolor="#fbe4d6" if not correcto else "#e2f5ec",
                                 edgecolor=COLOR_XGB if not correcto else COLOR_GREEN, linewidth=1.1))
    ax.text(7.85, 1.2, "Prueba", ha="center", va="center", fontsize=10)

    if correcto:
        # La flecha se separa claramente del borde inferior de las cajas (en vez de
        # correr casi pegada a él) y usa un verde más oscuro que el borde de la caja
        # "Prueba", para que ambos trazos no se confundan visualmente entre sí.
        ax.add_patch(FancyArrowPatch((3.0, 0.28), (7.85, 0.28), arrowstyle="-|>", mutation_scale=13,
                                      linewidth=1.6, color=COLOR_GREEN_DARK, connectionstyle="arc3,rad=0.0"))
        ax.text(5.4, -0.12, "estadísticos (media, mediana, escaladores) ajustados\nsolo aquí, aplicados hacia adelante",
                ha="center", fontsize=8.5, color=COLOR_GREEN_DARK, style="italic")
        ax.text(0.1, 2.3, "(b) Flujo correcto", fontsize=10.5, fontweight="bold", color=COLOR_GREEN)
    else:
        ax.add_patch(FancyArrowPatch((7.85, 2.05), (3.0, 2.05), arrowstyle="-|>", mutation_scale=13,
                                      linewidth=1.8, color=COLOR_RED, connectionstyle="arc3,rad=0.0"))
        ax.text(5.4, 2.35, "información del futuro filtrada hacia el pasado", ha="center",
                fontsize=8.5, color=COLOR_RED, style="italic")
        # Marca de "bloqueado" centrada en el hueco entre las dos cajas (x entre 5.7 y 6.0),
        # sobre la propia línea de la flecha, para que no se superponga con ningún borde ni texto.
        cx, cy, s = 5.85, 2.05, 0.12
        ax.plot([cx - s, cx + s], [cy - s, cy + s], color=COLOR_RED, linewidth=2.2, solid_capstyle="round", zorder=5)
        ax.plot([cx - s, cx + s], [cy + s, cy - s], color=COLOR_RED, linewidth=2.2, solid_capstyle="round", zorder=5)
        ax.text(0.1, 2.3, "(a) Fuga de datos", fontsize=10.5, fontweight="bold", color=COLOR_RED)

fig.tight_layout()
guardar(fig, "fig_fuga_datos")

# ---------------------------------------------------------------------------
# 5. Grid search vs. random search (Bergstra y Bengio, 2012)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.3))
rng = np.random.default_rng(5)

# Funcion de desempeno: solo el parametro importante (eje x) afecta el resultado
def desempeno(x):
    return np.exp(-((x - 6.5) ** 2) / 3.0)

x_curve = np.linspace(0, 10, 200)

ax = axes[0]
grid_vals = np.linspace(0.8, 9.2, 5)
gx, gy = np.meshgrid(grid_vals, grid_vals)
ax.plot(x_curve, desempeno(x_curve) * 8 + 1, color=COLOR_GRID, linewidth=6, alpha=0.5, zorder=0)
ax.scatter(gx.ravel(), gy.ravel(), color=COLOR_XGB, s=35, zorder=2)
ax.set_title("Grid search (25 evaluaciones)", loc="left", fontsize=9.0, fontweight="bold")
ax.set_xlabel("Hiperparámetro del modelo\nque sí influye en el desempeño (esquemático)", fontsize=7.8)
ax.set_ylabel("Hiperparámetro del modelo\nque no influye en el desempeño (esquemático)", fontsize=7.8)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.5, -0.62, f"Solo {len(set(np.round(grid_vals,1)))} valores distintos probados\nen el hiperparámetro relevante",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=COLOR_MUTED)

ax = axes[1]
rx = rng.uniform(0.3, 9.7, 25)
ry = rng.uniform(0.3, 9.7, 25)
ax.plot(x_curve, desempeno(x_curve) * 8 + 1, color=COLOR_GRID, linewidth=6, alpha=0.5, zorder=0)
ax.scatter(rx, ry, color=COLOR_LSTM, s=35, zorder=2)
ax.set_title("Random search (25 evaluaciones)", loc="left", fontsize=9.0, fontweight="bold")
ax.set_xlabel("Hiperparámetro del modelo\nque sí influye en el desempeño (esquemático)", fontsize=7.8)
ax.set_ylabel("Hiperparámetro del modelo\nque no influye en el desempeño (esquemático)", fontsize=7.8)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.5, -0.62, f"{len(set(np.round(rx,3)))} valores distintos probados\nen el hiperparámetro relevante",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=COLOR_MUTED)

fig.tight_layout()
guardar(fig, "fig_grid_vs_random_search")

# ---------------------------------------------------------------------------
# 6. La maldicion de la dimensionalidad: volumen relativo de una esfera inscrita
# ---------------------------------------------------------------------------
from scipy.special import gamma as gamma_fn

dims = np.arange(1, 21)
# Volumen de una bola unitaria de radio 0.5 (inscrita en un hipercubo de lado 1) en n dimensiones
r = 0.5
vol_esfera = (np.pi ** (dims / 2) / gamma_fn(dims / 2 + 1)) * r ** dims
vol_cubo = 1.0 ** dims
fraccion = vol_esfera / vol_cubo

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(dims, fraccion * 100, color=COLOR_XGB, linewidth=1.8, marker="o", markersize=3.5)
ax.set_yscale("log")
ax.set_xlabel("Número de dimensiones (variables) $d$")
ax.set_ylabel("Volumen de la esfera inscrita\nrespecto al hipercubo (\\%, escala log.)")
ax.axvline(14, color=COLOR_LSTM, linewidth=1.0, linestyle=":")
ax.text(14.3, fraccion[13] * 100 * 3, "$d=14$\n(variables del tensor\nde esta tesis)",
        fontsize=6.8, color=COLOR_LSTM)
fig.tight_layout()
guardar(fig, "fig_maldicion_dimensionalidad")

print("Segunda tanda de figuras conceptuales del marco teórico generada.")
