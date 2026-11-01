#!/usr/bin/env python3
"""
Genera figuras estilo publicación científica para la tesis (Capítulo IV).
Lee los datos exportados por el notebook en tesis/figuras_data/ y
escribe PDF (vectorial, para incluir en LaTeX) + PNG (previsualización) en
tesis/figuras/.

Paleta: validada con el skill dataviz (categórica, CVD-safe, par 1-2).
    LSTM     -> #2a78d6 (azul)
    XGBoost  -> #eb6834 (naranja)
    Observado/Real -> #1a1a1a (casi negro, referencia)
Tipografía: serif tipo STIX (aspecto de revista científica), sin requerir LaTeX del sistema.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = str(Path(__file__).resolve().parent.parent / "tesis")
DATA = f"{BASE}/figuras_data"
OUT = f"{BASE}/figuras"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Estilo global "revista científica"
# ---------------------------------------------------------------------------
COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_OBS = "#1a1a1a"
COLOR_GRID = "#d8d8d4"
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#52514e"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": COLOR_MUTED,
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
})

ESTACIONES = ['AUGUSTO_WEBERBAUER', 'ASUNCION', 'JESUS', 'LA_ENCANADA', 'NAMORA', 'SAN_JUAN']
ESTACIONES_LABEL = {
    'AUGUSTO_WEBERBAUER': 'A. Weberbauer', 'ASUNCION': 'Asunción', 'JESUS': 'Jesús',
    'LA_ENCANADA': 'La Encañada', 'NAMORA': 'Namora', 'SAN_JUAN': 'San Juan'
}

def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    fig.savefig(f"{OUT}/{nombre}.png")
    plt.close(fig)
    print(f"  -> {nombre}.pdf / .png")

# ---------------------------------------------------------------------------
# Figura 1: Matriz de correlación espacial
# ---------------------------------------------------------------------------
print("Figura 1: correlación espacial")
corr = pd.read_csv(f"{DATA}/corr_espacial.csv", index_col=0)
corr.index = [ESTACIONES_LABEL[e] for e in corr.index]
corr.columns = [ESTACIONES_LABEL[e] for e in corr.columns]

fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(corr.values, cmap="Blues", vmin=0.3, vmax=1.0)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=40, ha="right")
ax.set_yticklabels(corr.index)
ax.grid(False)
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        val = corr.values[i, j]
        color = "white" if val > 0.68 else COLOR_TEXT
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=color)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Coeficiente de correlación de Pearson", fontsize=8)
cbar.ax.tick_params(labelsize=7)
ax.set_title("Correlación espacial entre estaciones pluviométricas", loc="left", fontweight="bold")
guardar(fig, "fig_correlacion_espacial")

# ---------------------------------------------------------------------------
# Figura 2: Climatología anual regional
# ---------------------------------------------------------------------------
print("Figura 2: climatología regional")
clim = pd.read_csv(f"{DATA}/climatologia_regional.csv", index_col=0)
meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.bar(meses, clim["mean"], color=COLOR_LSTM, alpha=0.85, width=0.62,
       edgecolor="white", linewidth=0.5, zorder=3)
ax.errorbar(meses, clim["mean"], yerr=clim["std"], fmt="none", ecolor=COLOR_TEXT,
            elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
ax.set_ylabel("Precipitación acumulada mensual (mm)")
ax.set_title("Ciclo climatológico anual medio regional (2000-2025)", loc="left", fontweight="bold")
ax.set_axisbelow(True)
guardar(fig, "fig_climatologia_regional")

# ---------------------------------------------------------------------------
# Figura 3: Dispersión OBS vs PISCO (grilla 3x2)
# ---------------------------------------------------------------------------
print("Figura 3: dispersión OBS vs PISCO")
from sklearn.metrics import r2_score
df_op = pd.read_csv(f"{DATA}/obs_vs_pisco.csv", index_col=0)

fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.4))
axes = axes.flatten()
for i, est in enumerate(ESTACIONES):
    ax = axes[i]
    x = df_op[f"{est}_OBS"].values
    y = df_op[f"{est}_PISCO"].values
    r2 = r2_score(x, y)
    ax.scatter(x, y, s=4, alpha=0.18, color=COLOR_LSTM, edgecolor="none", rasterized=True)
    maxv = max(x.max(), y.max())
    ax.plot([0, maxv], [0, maxv], color=COLOR_XGB, linestyle="--", linewidth=1.1)
    ax.set_title(f"{ESTACIONES_LABEL[est]} ($R^2$={r2:.2f})", fontsize=8.5)
    ax.set_xlabel("Observado (mm)", fontsize=7.5)
    ax.set_ylabel("PISCO (mm)", fontsize=7.5)
fig.suptitle("Dispersión Observado vs. PISCO por estación (diario, 2000-2025)", fontweight="bold", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
guardar(fig, "fig_dispersion_obs_pisco")

# ---------------------------------------------------------------------------
# Figura 4: Curva de convergencia LSTM
# ---------------------------------------------------------------------------
print("Figura 4: convergencia LSTM")
hist = pd.read_csv(f"{DATA}/historial_lstm_18var.csv")
mejor_epoca = int(hist["val_loss"].idxmin()) + 1

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(hist["epoca"] + 1, hist["loss"], color=COLOR_LSTM, linewidth=1.4, label="Entrenamiento")
ax.plot(hist["epoca"] + 1, hist["val_loss"], color=COLOR_XGB, linewidth=1.4, linestyle="--", label="Validación")
ax.axvline(mejor_epoca, color=COLOR_MUTED, linestyle=":", linewidth=1.0)
ax.text(mejor_epoca, ax.get_ylim()[1]*0.95, f" época óptima {mejor_epoca}", fontsize=7, color=COLOR_MUTED, va="top")
ax.set_xlabel("Época de entrenamiento (número de veces)")
ax.set_ylabel("Error cuadrático asimétrico")
ax.set_title("Curva de convergencia — Red LSTM", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.tight_layout()
guardar(fig, "fig_convergencia_lstm")

# ---------------------------------------------------------------------------
# Figura 5: Importancia de variables XGBoost (Top 15)
# ---------------------------------------------------------------------------
print("Figura 5: importancia de variables")
imp = np.load(f"{DATA}/importancia_xgb_18var.npz", allow_pickle=True)
importancias, nombres = imp["importancias"], imp["nombres"]
idx_top = np.argsort(importancias)[-15:]
nombres_top = [str(n) for n in nombres[idx_top]]
val_top = importancias[idx_top]

COLOR_CLIMA = "#9b59b6"

def clasificar(n):
    if "PISCO" in n:
        return COLOR_XGB
    if "Estacional" in n:
        return "#1baf7a"
    if "humedad_relativa" in n or "temperature_2m_min" in n or "u_component_of_wind" in n or "v_component_of_wind" in n:
        return COLOR_CLIMA
    return COLOR_LSTM

colores = [clasificar(n) for n in nombres_top]

fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.barh(range(len(nombres_top)), val_top, color=colores, edgecolor="white", linewidth=0.4)
ax.set_yticks(range(len(nombres_top)))
ax.set_yticklabels(nombres_top, fontsize=7.5)
ax.set_xlabel("Ganancia de información relativa (F-score)")
ax.set_title("Top 15 variables — XGBoost (A. Weberbauer)", loc="left", fontweight="bold")
from matplotlib.patches import Patch
leyenda = [Patch(facecolor=COLOR_LSTM, label="Memoria terrestre"),
           Patch(facecolor=COLOR_XGB, label="Forzamiento PISCO"),
           Patch(facecolor=COLOR_CLIMA, label="Variables climáticas"),
           Patch(facecolor="#1baf7a", label="Reloj estacional")]
ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout()
guardar(fig, "fig_importancia_variables")

# ---------------------------------------------------------------------------
# Figura 6: Hietograma comparativo + curva de excedencia (A. Weberbauer)
# ---------------------------------------------------------------------------
print("Figura 6: hietograma y curva de excedencia")
series = np.load(f"{DATA}/test_series_fisicas_18var.npz", allow_pickle=True)
obs, pred_lstm, pred_xgb = series["obs"], series["pred_lstm"], series["pred_xgb"]
idx_est = 0
obs_real = obs[:, idx_est]; sim_lstm = pred_lstm[:, idx_est]; sim_xgb = pred_xgb[:, idx_est]

fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.0))
inicio, fin = 150, 250
axes[0].plot(obs_real[inicio:fin], color=COLOR_OBS, linewidth=1.3, label="Observado")
axes[0].plot(sim_lstm[inicio:fin], color=COLOR_LSTM, linewidth=1.1, label="LSTM")
axes[0].plot(sim_xgb[inicio:fin], color=COLOR_XGB, linewidth=1.1, linestyle="--", label="XGBoost")
axes[0].set_ylabel("Precipitación (mm/día)")
axes[0].set_xlabel("Horizonte de prueba (días)")
axes[0].set_title("(a) Segmento de hietograma predictivo — A. Weberbauer", loc="left", fontweight="bold", fontsize=9)
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

obs_ord = np.sort(obs_real)[::-1]; lstm_ord = np.sort(sim_lstm)[::-1]; xgb_ord = np.sort(sim_xgb)[::-1]
prob = np.arange(1, len(obs_ord)+1) / len(obs_ord) * 100
lim = int(len(prob) * 0.20)
axes[1].plot(prob[:lim], obs_ord[:lim], color=COLOR_OBS, linewidth=1.3, label="Observado")
axes[1].plot(prob[:lim], lstm_ord[:lim], color=COLOR_LSTM, linewidth=1.1, label="LSTM")
axes[1].plot(prob[:lim], xgb_ord[:lim], color=COLOR_XGB, linewidth=1.1, linestyle="--", label="XGBoost")
axes[1].set_yscale("log")
axes[1].set_xlabel("Probabilidad de excedencia (%)")
axes[1].set_ylabel("Magnitud del evento (mm/día)")
axes[1].set_title("(b) Curva de probabilidad de excedencia (eventos extremos)", loc="left", fontweight="bold", fontsize=9)
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
fig.tight_layout()
guardar(fig, "fig_hietograma_excedencia")

# ---------------------------------------------------------------------------
# Figura 7: Comparación RMSE/NSE por estación (LSTM vs XGBoost)
# LSTM: promedio ± desviación estándar de 10 inicializaciones independientes.
# XGBoost: determinista (random_state fijo), valor único.
# ---------------------------------------------------------------------------
print("Figura 7: comparación de métricas por estación (10 semillas)")
met = pd.read_csv(f"{DATA}/multisemilla_18var_resumen.csv")
met = met.set_index("Estación").reindex(ESTACIONES)
labels = [ESTACIONES_LABEL[e] for e in ESTACIONES]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
x = np.arange(len(labels)); w = 0.36

axes[0].bar(x - w/2, met["RMSE_LSTM_media"], width=w, yerr=met["RMSE_LSTM_std"],
            color=COLOR_LSTM, label="LSTM", capsize=3,
            error_kw=dict(elinewidth=1.0, ecolor=COLOR_MUTED))
axes[0].bar(x + w/2, met["RMSE_XGBoost"], width=w, color=COLOR_XGB, label="XGBoost")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=35, ha="right")
axes[0].set_ylabel("RMSE (mm/día)")
axes[0].set_title("(a) RMSE por estación", loc="left", fontweight="bold", fontsize=9)
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2)

axes[1].bar(x - w/2, met["NSE_LSTM_media"], width=w, yerr=met["NSE_LSTM_std"],
            color=COLOR_LSTM, label="LSTM", capsize=3,
            error_kw=dict(elinewidth=1.0, ecolor=COLOR_MUTED))
axes[1].bar(x + w/2, met["NSE_XGBoost"], width=w, color=COLOR_XGB, label="XGBoost")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=35, ha="right")
axes[1].set_ylabel("NSE (adimensional)")
axes[1].set_title("(b) NSE por estación", loc="left", fontweight="bold", fontsize=9)
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2)
fig.tight_layout()
guardar(fig, "fig_comparacion_metricas")

# ---------------------------------------------------------------------------
# Figura 8: Escenarios de vacíos simulados
# ---------------------------------------------------------------------------
print("Figura 8: escenarios de vacíos simulados")
esc = pd.read_csv(f"{DATA}/escenarios_vacios_18var.csv")
resumen = esc.groupby(["Escenario", "Modelo"])[["RMSE", "NSE"]].mean().reset_index()
orden_esc = ["A (2%)", "B (5%)", "C (8%)", "D (10%)"]
resumen["Escenario"] = pd.Categorical(resumen["Escenario"], categories=orden_esc, ordered=True)
resumen = resumen.sort_values("Escenario")

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
for modelo, color, marker in [("LSTM", COLOR_LSTM, "o"), ("XGBoost", COLOR_XGB, "s")]:
    sub = resumen[resumen["Modelo"] == modelo]
    axes[0].plot(sub["Escenario"].astype(str), sub["NSE"], marker=marker, color=color, linewidth=1.4, label=modelo)
    axes[1].plot(sub["Escenario"].astype(str), sub["RMSE"], marker=marker, color=color, linewidth=1.4, label=modelo)
axes[0].set_ylabel("NSE promedio (6 estaciones)")
axes[0].set_title("(a) NSE vs. severidad del vacío simulado", loc="left", fontweight="bold", fontsize=9)
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
axes[1].set_ylabel("RMSE promedio (mm/día)")
axes[1].set_title("(b) RMSE vs. severidad del vacío simulado", loc="left", fontweight="bold", fontsize=9)
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.tight_layout()
guardar(fig, "fig_escenarios_vacios")

print("\nTodas las figuras generadas en:", OUT)
