#!/usr/bin/env python3
"""
Genera el análisis exploratorio de datos (EDA) del registro crudo de SENAMHI
para la tesis: estadísticos descriptivos por estación, serie temporal de 26
años y curva de doble masa. Lee directamente datos/DATA_PP.csv (sin pasar por
el notebook) y escribe PDF+PNG en tesis/figuras/ y un CSV de estadísticos en
tesis/figuras_data/.

Mismo tratamiento de códigos no numéricos que el notebook de procesamiento:
'S/D' (sin dato) -> NaN, 'T' (trazas) -> 0.0.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

BASE = str(Path(__file__).resolve().parent.parent)
DATOS = f"{BASE}/datos/DATA_PP.csv"
OUT = f"{BASE}/tesis/figuras"
OUT_DATA = f"{BASE}/tesis/figuras_data"
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)

COLOR_OBS = "#1a1a1a"
COLOR_GRID = "#d8d8d4"
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#52514e"
COLOR_SERIE = "#2a78d6"
COLOR_TENDENCIA = "#eb6834"

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
# Carga y limpieza (misma lógica que el notebook de procesamiento)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATOS, sep=';')
df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%y')
df = df.set_index('FECHA')

CODIGOS_CONOCIDOS = {'S/D': np.nan, 'T': 0.0}
for col in ESTACIONES:
    df[col] = df[col].replace(CODIGOS_CONOCIDOS)
    df[col] = pd.to_numeric(df[col], errors='raise')

# ---------------------------------------------------------------------------
# Tabla 1: estadísticos descriptivos por estación
# ---------------------------------------------------------------------------
print("Tabla: estadísticos descriptivos por estación")
dias_lluviosos = (df[ESTACIONES] > 0.1).sum()
n_validos = df[ESTACIONES].notna().sum()

stats = pd.DataFrame({
    'media': df[ESTACIONES].mean(),
    'desv_std': df[ESTACIONES].std(),
    'maximo': df[ESTACIONES].max(),
    'p50': df[ESTACIONES].median(),
    'p95': df[ESTACIONES].quantile(0.95),
    'p99': df[ESTACIONES].quantile(0.99),
    'pct_dias_secos': 100 * (1 - dias_lluviosos / n_validos),
})
stats.index.name = 'estacion'
stats.to_csv(f"{OUT_DATA}/estadisticos_descriptivos_senamhi.csv")
print(stats.round(2))

# ---------------------------------------------------------------------------
# Figura: serie temporal diaria, 26 años, grilla 3x2
# ---------------------------------------------------------------------------
print("Figura: serie temporal 2000-2025")
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(11, 15), sharex=True, sharey=True)
for i, est in enumerate(ESTACIONES):
    ax = axes[i]
    # NaN (vacíos "S/D") se dejan como huecos: no se interpola ni se conecta la línea.
    ax.plot(df.index, df[est], color=COLOR_SERIE, linewidth=0.35)
    ax.set_title(ESTACIONES_LABEL[est], loc="left", fontweight="bold", fontsize=9)
    ax.set_ylabel("PP diaria (mm)")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelbottom=True, labelsize=7, rotation=90)
    ax.set_xlabel("Año", fontsize=8)
    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2025-12-31"))

fig.suptitle("Serie temporal de precipitación diaria registrada por SENAMHI (2000-2025)",
             fontweight="bold", fontsize=12, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.985])
guardar(fig, "fig_serie_temporal_senamhi")

# ---------------------------------------------------------------------------
# Figura: curva de doble masa, grilla 3x2
# ---------------------------------------------------------------------------
print("Figura: curva de doble masa")
df_annual = df[ESTACIONES].resample('YE').sum(min_count=1)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 10.5))
axes = axes.flatten()
for i, est in enumerate(ESTACIONES):
    ax = axes[i]
    patron = [e for e in ESTACIONES if e != est]
    promedio_patron_anual = df_annual[patron].mean(axis=1)
    acumulado_patron = promedio_patron_anual.cumsum()
    acumulado_estacion = df_annual[est].cumsum()

    valid = acumulado_patron.notna() & acumulado_estacion.notna()
    x = acumulado_patron[valid].values.reshape(-1, 1)
    y = acumulado_estacion[valid].values

    ax.plot(acumulado_patron, acumulado_estacion, marker='o', markersize=4,
            linestyle='-', linewidth=1, color=COLOR_SERIE, label='Acumulado observado')

    if len(x) > 1:
        modelo = LinearRegression().fit(x, y)
        r2 = modelo.score(x, y)
        ax.plot(x, modelo.predict(x), color=COLOR_TENDENCIA, linestyle='--',
                 linewidth=1.3, label=f'Tendencia ($R^2$={r2:.3f})')

    ax.set_title(ESTACIONES_LABEL[est], loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel("PP acum. estaciones patrón (mm)", fontsize=7.5)
    ax.set_ylabel("PP acum. estación (mm)", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=6.5, frameon=True)

fig.suptitle("Curva de doble masa: consistencia temporal del registro (acumulado anual, 2000-2025)",
             fontweight="bold", fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
guardar(fig, "fig_doble_masa")

print("\nListo.")
