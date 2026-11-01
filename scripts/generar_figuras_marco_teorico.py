#!/usr/bin/env python3
"""
Genera 2 figuras conceptuales ORIGINALES (no descargadas de internet, sin
problema de licencia) para el Marco Teorico: concepto de Gradient Boosting
secuencial y concepto de imputacion multivariada con datos correlacionados.
Mismo estilo que generar_figuras.py.
"""
import numpy as np
import matplotlib.pyplot as plt

OUT = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
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
# Concepto: Gradient Boosting secuencial (funcion objetivo real -> residuos)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
x = np.linspace(0, 10, 60)
y_real = 3*np.sin(x*0.8) + 0.15*x**1.4 + rng.normal(0, 0.4, len(x))

def arbol_tosco(x, y, n_hojas):
    bordes = np.linspace(x.min(), x.max(), n_hojas+1)
    pred = np.zeros_like(y)
    for i in range(n_hojas):
        m = (x >= bordes[i]) & (x <= bordes[i+1] + 1e-9)
        if m.sum() > 0:
            pred[m] = y[m].mean()
    return pred

fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=False)
residuo = y_real.copy()
pred_acumulada = np.zeros_like(y_real)
titulos = ["Árbol 1: ajusta $y$", "Árbol 2: ajusta el residuo de (1)", "Árbol 3: ajusta el residuo de (1)+(2)"]
for i, ax in enumerate(axes):
    arbol_i = arbol_tosco(x, residuo, n_hojas=3 + i)
    pred_acumulada = pred_acumulada + arbol_i
    ax.plot(x, residuo, "o", color=COLOR_MUTED, markersize=3, alpha=0.55, label="Objetivo actual")
    ax.step(x, arbol_i, color=COLOR_XGB, linewidth=1.6, where="mid", label=f"Árbol {i+1}")
    ax.axhline(0, color=COLOR_GRID, linewidth=0.8)
    ax.set_title(titulos[i], fontsize=8.3)
    ax.set_xlabel("$x$")
    if i == 0:
        ax.set_ylabel("$y$ / residuo")
    residuo = residuo - arbol_i
fig.suptitle("Concepto de Gradient Boosting: ajuste secuencial de residuos", fontsize=10, fontweight="bold", x=0.02, ha="left")
h, _ = axes[0].get_legend_handles_labels()
fig.legend(h, ["Objetivo actual ($y$ o residuo)", "Ajuste del árbol de cada panel"], loc="lower center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=[0, 0.07, 1, 0.92])
guardar(fig, "fig_concepto_boosting")

# ---------------------------------------------------------------------------
# Concepto: Imputacion multivariada usando una estacion vecina correlacionada
# ---------------------------------------------------------------------------
rng = np.random.default_rng(3)
t = np.arange(60)
base = 4 + 3*np.sin(t/6) + rng.normal(0, 0.6, len(t))
vecina = 0.8*base + rng.normal(0, 0.5, len(t)) + 1.0
vecina = np.clip(vecina, 0, None)

serie_con_vacios = base.copy()
mask_vacio = (t >= 22) & (t <= 33)
serie_con_vacios[mask_vacio] = np.nan

# "Imputacion": regresion lineal simple entrenada en tramos NO faltantes usando la vecina
obs_idx = ~mask_vacio
coef = np.polyfit(vecina[obs_idx], base[obs_idx], 1)
imputado = np.polyval(coef, vecina)

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(t, vecina, color=COLOR_MUTED, linewidth=1.0, linestyle=":", label="Estación vecina (predictor)")
ax.plot(t, serie_con_vacios, color=COLOR_OBS, linewidth=1.5, label="Estación con vacío (observado)")
ax.plot(t[mask_vacio], imputado[mask_vacio], color=COLOR_XGB, linewidth=1.8, label="Tramo reconstruido (imputado)")
ax.axvspan(22, 33, color=COLOR_XGB, alpha=0.08)
ax.set_xlabel("Día del registro pluviométrico (esquemático)")
ax.set_ylabel("Precipitación (mm/día, esquemático)")
ax.set_title("Concepto de imputación multivariada usando correlación espacial", loc="left", fontweight="bold", fontsize=9.5)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.5)
fig.tight_layout()
guardar(fig, "fig_concepto_imputacion")

print("Figuras conceptuales generadas.")
