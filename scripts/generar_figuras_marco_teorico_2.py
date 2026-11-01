#!/usr/bin/env python3
"""Figuras conceptuales adicionales: funciones de activacion, particion
cronologica train/val/test, efecto del escalado, y precipitacion vs altitud."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = str(Path(__file__).resolve().parent.parent)
DATA = f"{BASE}/tesis/figuras_data"
OUT = f"{BASE}/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#d8d8d4"
COLOR_GREEN = "#1baf7a"

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
# Figura: funciones de activacion
# ---------------------------------------------------------------------------
z = np.linspace(-6, 6, 400)
sigmoide = 1 / (1 + np.exp(-z))
tanh_ = np.tanh(z)
relu = np.maximum(0, z)

fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), sharex=True)
for ax, y, nombre, color in zip(axes, [sigmoide, tanh_, relu],
                                  [r"Sigmoide $\sigma(z)$", r"Tangente hiperbólica $\tanh(z)$", "ReLU"],
                                  [COLOR_LSTM, COLOR_XGB, COLOR_GREEN]):
    ax.plot(z, y, color=color, linewidth=1.8)
    ax.axhline(0, color=COLOR_GRID, linewidth=0.7)
    ax.axvline(0, color=COLOR_GRID, linewidth=0.7)
    ax.set_title(nombre, fontsize=9)
    ax.set_xlabel("Valor de entrada a la función ($z$)")
axes[0].set_ylabel("Valor de salida de la función")
fig.suptitle("Funciones de activación empleadas en la LSTM y en la capa de salida", fontsize=10, fontweight="bold", x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.90])
guardar(fig, "fig_funciones_activacion")

# ---------------------------------------------------------------------------
# Figura: linea de tiempo de la particion train/val/test
# ---------------------------------------------------------------------------
fechas = pd.date_range("2000-01-01", "2025-12-31", freq="D")
n = len(fechas)
idx_train = int(n * 0.70)
idx_val = int(n * 0.85)

fig, ax = plt.subplots(figsize=(8.2, 1.9))
ax.barh(0, fechas[idx_train] - fechas[0], left=fechas[0], height=0.55, color=COLOR_LSTM, label="Entrenamiento (70%)")
ax.barh(0, fechas[idx_val] - fechas[idx_train], left=fechas[idx_train], height=0.55, color=COLOR_GREEN, label="Validación (15%)")
ax.barh(0, fechas[-1] - fechas[idx_val], left=fechas[idx_val], height=0.55, color=COLOR_XGB, label="Prueba (15%)")

for x, label in [(fechas[0], fechas[0].strftime("%Y")), (fechas[idx_train], fechas[idx_train].strftime("%Y")),
                  (fechas[idx_val], fechas[idx_val].strftime("%Y")), (fechas[-1], fechas[-1].strftime("%Y"))]:
    ax.axvline(x, color="white", linewidth=1.2, zorder=3)

ax.set_yticks([])
ax.set_xlim(fechas[0], fechas[-1])
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title("Partición cronológica estricta del tensor de secuencias (sin mezcla aleatoria)", loc="left", fontweight="bold", fontsize=9.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
ax.grid(False)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
guardar(fig, "fig_linea_tiempo_particion")

# ---------------------------------------------------------------------------
# Figura: efecto del escalado (Robust + MinMax) sobre datos reales
# ---------------------------------------------------------------------------
from sklearn.preprocessing import RobustScaler, MinMaxScaler
df_op = pd.read_csv(f"{DATA}/obs_vs_pisco.csv", index_col=0)
serie = df_op["AUGUSTO_WEBERBAUER_OBS"].dropna().values.reshape(-1, 1)

idx_split = int(len(serie) * 0.70)
robust = RobustScaler().fit(serie[:idx_split])
serie_robust = robust.transform(serie)
minmax = MinMaxScaler().fit(serie_robust[:idx_split])
serie_final = minmax.transform(serie_robust)

fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.2))
for ax, data, titulo, xlabel, color in zip(
    axes, [serie.flatten(), serie_robust.flatten(), serie_final.flatten()],
    ["Original (mm/día)", "Tras RobustScaler", "Tras MinMaxScaler (final)"],
    ["Precipitación (mm/día)", "Valor tras RobustScaler (adimensional)", "Valor tras MinMaxScaler [0, 1]"],
    [COLOR_MUTED, COLOR_LSTM, COLOR_XGB]
):
    ax.hist(data, bins=40, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_title(titulo, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=7.6)
    ax.set_ylabel("Frecuencia" if ax is axes[0] else "")
fig.suptitle("Efecto del escalado en dos etapas sobre la distribución de precipitación (A. Weberbauer)",
             fontsize=9.7, fontweight="bold", x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.90])
guardar(fig, "fig_efecto_escalado")

# ---------------------------------------------------------------------------
# Figura: precipitacion anual media vs altitud de la estacion
# ---------------------------------------------------------------------------
ESTACIONES = ['AUGUSTO_WEBERBAUER', 'ASUNCION', 'JESUS', 'LA_ENCANADA', 'NAMORA', 'SAN_JUAN']
ESTACIONES_LABEL = {
    'AUGUSTO_WEBERBAUER': 'A. Weberbauer', 'ASUNCION': 'Asunción', 'JESUS': 'Jesús',
    'LA_ENCANADA': 'La Encañada', 'NAMORA': 'Namora', 'SAN_JUAN': 'San Juan'
}
ALTITUDES = {'AUGUSTO_WEBERBAUER': 2673, 'ASUNCION': 2270, 'JESUS': 2564,
             'LA_ENCANADA': 2980, 'NAMORA': 2760, 'SAN_JUAN': 2253}

medias_anuales = {}
for est in ESTACIONES:
    col = f"{est}_OBS"
    serie_est = df_op[col].dropna()
    serie_est.index = pd.to_datetime(serie_est.index)
    medias_anuales[est] = serie_est.resample("YE").sum().mean()

fig, ax = plt.subplots(figsize=(5.6, 4.0))
for est in ESTACIONES:
    ax.scatter(ALTITUDES[est], medias_anuales[est], s=70, color=COLOR_LSTM, edgecolor="white", zorder=3)
    ax.annotate(f"{ESTACIONES_LABEL[est]}", (ALTITUDES[est], medias_anuales[est]), fontsize=7.8,
                xytext=(6, 5), textcoords="offset points", color=COLOR_MUTED)

# Linea de tendencia simple
coef = np.polyfit(list(ALTITUDES.values()), [medias_anuales[e] for e in ESTACIONES], 1)
xs = np.linspace(min(ALTITUDES.values()) - 50, max(ALTITUDES.values()) + 50, 10)
ax.plot(xs, np.polyval(coef, xs), color=COLOR_XGB, linestyle="--", linewidth=1.3,
        label=f"Tendencia lineal ({coef[0]:.2f} mm/año por metro)")

ax.set_xlabel("Altitud de la estación (msnm)")
ax.set_ylabel("Precipitación anual media (mm/año)")
ax.set_title("Precipitación anual media vs. altitud de la estación", loc="left", fontweight="bold", fontsize=9.5)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=7.5)
ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
fig.tight_layout()
guardar(fig, "fig_precipitacion_altitud")

print("Listo.")
