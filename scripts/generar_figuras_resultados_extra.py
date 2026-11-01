#!/usr/bin/env python3
"""Figuras adicionales para ampliar el Capitulo IV: boxplot de residuos y
dispersion de caracteristicas de estacion vs NSE final."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = str(Path(__file__).resolve().parent.parent / "tesis")
DATA = f"{BASE}/figuras_data"
OUT = f"{BASE}/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#d8d8d4"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": COLOR_MUTED, "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a", "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED,
    "axes.grid": True, "grid.color": COLOR_GRID, "grid.linewidth": 0.6,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
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
# Figura: boxplot de residuos (Pred - Obs) por estacion y modelo
# ---------------------------------------------------------------------------
series = np.load(f"{DATA}/test_series_fisicas_18var.npz", allow_pickle=True)
obs, pred_lstm, pred_xgb = series["obs"], series["pred_lstm"], series["pred_xgb"]

filas = []
for idx, est in enumerate(ESTACIONES):
    res_lstm = pred_lstm[:, idx] - obs[:, idx]
    res_xgb = pred_xgb[:, idx] - obs[:, idx]
    for v in res_lstm:
        filas.append({"Estación": ESTACIONES_LABEL[est], "Modelo": "LSTM", "Residuo": v})
    for v in res_xgb:
        filas.append({"Estación": ESTACIONES_LABEL[est], "Modelo": "XGBoost", "Residuo": v})
df_res = pd.DataFrame(filas)

fig, ax = plt.subplots(figsize=(8.6, 4.2))
posiciones = np.arange(len(ESTACIONES))
ancho = 0.32
datos_lstm = [df_res[(df_res["Estación"]==ESTACIONES_LABEL[e]) & (df_res["Modelo"]=="LSTM")]["Residuo"].values for e in ESTACIONES]
datos_xgb = [df_res[(df_res["Estación"]==ESTACIONES_LABEL[e]) & (df_res["Modelo"]=="XGBoost")]["Residuo"].values for e in ESTACIONES]

bp1 = ax.boxplot(datos_lstm, positions=posiciones - ancho/2 - 0.02, widths=ancho, patch_artist=True,
                  showfliers=False, medianprops=dict(color="black", linewidth=1.2))
bp2 = ax.boxplot(datos_xgb, positions=posiciones + ancho/2 + 0.02, widths=ancho, patch_artist=True,
                  showfliers=False, medianprops=dict(color="black", linewidth=1.2))
for box in bp1['boxes']:
    box.set(facecolor=COLOR_LSTM, alpha=0.75, edgecolor=COLOR_MUTED)
for box in bp2['boxes']:
    box.set(facecolor=COLOR_XGB, alpha=0.75, edgecolor=COLOR_MUTED)

ax.axhline(0, color=COLOR_MUTED, linewidth=0.9, linestyle="--")
ax.set_xticks(posiciones)
ax.set_xticklabels([ESTACIONES_LABEL[e] for e in ESTACIONES], rotation=20, ha="right")
ax.set_ylabel("Residuo: Predicho $-$ Observado (mm/día)")
ax.set_title("Distribución del error residual por estación y modelo (conjunto de prueba)", loc="left", fontweight="bold", fontsize=9.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=COLOR_LSTM, alpha=0.75, label="LSTM"),
                    Patch(facecolor=COLOR_XGB, alpha=0.75, label="XGBoost")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
fig.tight_layout()
guardar(fig, "fig_boxplot_residuos")

# ---------------------------------------------------------------------------
# Figura: dispersion de caracteristicas de estacion vs NSE final (LSTM)
# ---------------------------------------------------------------------------
faltantes = {'AUGUSTO_WEBERBAUER':138,'ASUNCION':779,'JESUS':234,'LA_ENCANADA':62,'NAMORA':107,'SAN_JUAN':277}
pct_faltante = {k: v/9497*100 for k,v in faltantes.items()}
r2_pisco = {'AUGUSTO_WEBERBAUER':0.7615,'ASUNCION':0.7729,'JESUS':0.7262,'LA_ENCANADA':0.8263,'NAMORA':0.5486,'SAN_JUAN':0.7248}

met = pd.read_csv(f"{DATA}/metricas_finales_18var.csv")
nse_lstm = met[met.Modelo=='LSTM'].set_index('Estación')['NSE'].to_dict()

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
for est in ESTACIONES:
    axes[0].scatter(pct_faltante[est], nse_lstm[est], s=55, color=COLOR_LSTM, edgecolor="white", zorder=3)
    axes[0].annotate(ESTACIONES_LABEL[est], (pct_faltante[est], nse_lstm[est]), fontsize=7,
                      xytext=(4, 4), textcoords="offset points", color=COLOR_MUTED)
    axes[1].scatter(r2_pisco[est], nse_lstm[est], s=55, color=COLOR_XGB, edgecolor="white", zorder=3)
    axes[1].annotate(ESTACIONES_LABEL[est], (r2_pisco[est], nse_lstm[est]), fontsize=7,
                      xytext=(4, 4), textcoords="offset points", color=COLOR_MUTED)

axes[0].set_xlabel("Datos faltantes en el registro original (%)")
axes[0].set_ylabel("NSE (LSTM, conjunto de prueba)")
axes[0].set_title("(a) % faltante vs. NSE final", loc="left", fontweight="bold", fontsize=9)

axes[1].set_xlabel("$R^2$ Observado-PISCO")
axes[1].set_ylabel("NSE (LSTM, conjunto de prueba)")
axes[1].set_title("(b) Ajuste PISCO vs. NSE final", loc="left", fontweight="bold", fontsize=9)
fig.tight_layout()
guardar(fig, "fig_caracteristicas_vs_nse")

print("Listo.")
