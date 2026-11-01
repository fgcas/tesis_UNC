#!/usr/bin/env python3
"""
Figuras conceptuales para la tercera ampliación del Marco Teórico (Bases teóricas):
reanálisis atmosférico (asimilación de datos), teledetección pasiva vs. activa,
humedad relativa / punto de rocío (curva de saturación), y análisis de
sensibilidad (barrido de un parámetro, con los datos reales del análisis de
ventana ya realizado en esta tesis). Mismo estilo que los scripts anteriores.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras"
DATA = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras_data"

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
# 1. Reanálisis atmosférico: asimilación de datos (observaciones + modelo
#    numérico -> estado reanalizado), con el flujo de extracción vía GEE.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.6)
ax.axis("off")

cajas = [
    (0.3, 5.3, 2.5, 1.35, "Observaciones\nhistóricas\n(satélites, estaciones,\nradiosondeos, boyas)", COLOR_MUTED),
    (0.3, 2.3, 2.5, 1.35, "Modelo numérico\nde predicción\natmosférica", COLOR_MUTED),
    (3.8, 3.8, 2.6, 1.35, "Asimilación\nde datos\n(ciclo iterativo)", COLOR_GREEN_DARK),
    (7.2, 3.8, 2.5, 1.35, "ERA5-Land\n(estado reanalizado,\n0.1°, 1950-presente)", COLOR_LSTM),
]
for x, y, w, h, texto, color in cajas:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                                 facecolor="white", edgecolor=color, linewidth=1.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=7.8,
            color="#1a1a1a", fontweight="bold", zorder=4)

flechas = [
    ((2.8, 5.75), (3.8, 4.95)),
    ((2.8, 3.15), (3.8, 4.15)),
    ((6.4, 4.475), (7.2, 4.475)),
]
for (x0, y0), (x1, y1) in flechas:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                  color=COLOR_OBS, linewidth=1.3, zorder=2))

ax.text(5.0, 1.55, "Un reanálisis combina un modelo numérico con observaciones históricas mediante\n"
                   "asimilación de datos, para reconstruir el estado pasado de la atmósfera de forma\n"
                   "físicamente consistente y con cobertura global continua, sin los vacíos de una red de\n"
                   "estaciones puntual ni el error de un único sensor satelital aislado.",
        ha="center", va="top", fontsize=7.6, color=COLOR_MUTED, style="italic")

fig.tight_layout()
guardar(fig, "fig_reanalisis_asimilacion")

# ---------------------------------------------------------------------------
# 2. Teledetección: sensor pasivo vs. sensor activo
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4))

for ax, titulo in [(axes[0], "(a) Sensor pasivo\n(radiómetro de microondas, p. ej. GPM)"),
                    (axes[1], "(b) Sensor activo\n(radar, p. ej. precipitation radar)")]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title(titulo, fontsize=8.6, fontweight="bold", loc="center", pad=10)
    # satelite
    ax.add_patch(FancyBboxPatch((3.8, 8.4), 2.4, 1.1, boxstyle="round,pad=0.04,rounding_size=0.06",
                                 facecolor="white", edgecolor=COLOR_MUTED, linewidth=1.4, zorder=3))
    ax.text(5.0, 8.95, "Satélite", ha="center", va="center", fontsize=7.8, fontweight="bold", zorder=4)
    # nube/lluvia
    ax.add_patch(FancyBboxPatch((3.3, 3.6), 3.4, 1.7, boxstyle="round,pad=0.05,rounding_size=0.3",
                                 facecolor="#eef3fa", edgecolor=COLOR_GRID, linewidth=1.0, zorder=3))
    ax.text(5.0, 4.45, "Nube / lluvia", ha="center", va="center", fontsize=7.6, color=COLOR_MUTED, zorder=4)

# Pasivo: flechas de abajo hacia arriba (radiacion emitida/dispersada por la lluvia)
ax = axes[0]
for dx in [-1.0, 0, 1.0]:
    ax.add_patch(FancyArrowPatch((5.0 + dx, 5.35), (5.0 + dx * 1.4, 8.35), arrowstyle="-|>",
                                  mutation_scale=11, color=COLOR_LSTM, linewidth=1.3, zorder=2))
ax.text(5.0, 1.9, "El sensor recibe la radiación de\nmicroondas que la lluvia ya emite\no dispersa de forma natural",
        ha="center", va="center", fontsize=7.2, color=COLOR_MUTED)

# Activo: flecha de ida (emision) y de vuelta (retorno/backscatter), bien separadas
ax = axes[1]
ax.add_patch(FancyArrowPatch((4.1, 8.35), (4.1, 5.4), arrowstyle="-|>",
                              mutation_scale=12, color=COLOR_XGB, linewidth=1.4, zorder=2))
ax.add_patch(FancyArrowPatch((6.3, 5.4), (6.3, 8.35), arrowstyle="-|>",
                              mutation_scale=12, color=COLOR_XGB, linewidth=1.4, linestyle="--", zorder=2))
ax.text(3.55, 6.85, "emite", ha="center", va="center", fontsize=7.4, color=COLOR_XGB, rotation=90)
ax.text(6.85, 6.85, "recibe\n(backscatter)", ha="center", va="center", fontsize=7.4, color=COLOR_XGB, rotation=90)
ax.text(5.0, 1.9, "El sensor emite su propia señal de\nradar y mide la energía reflejada\nde vuelta por las gotas de lluvia",
        ha="center", va="center", fontsize=7.2, color=COLOR_MUTED)

leyenda_handles = [
    mpatches.Patch(facecolor=COLOR_LSTM, label="Radiación recibida (sensor pasivo)"),
    mpatches.Patch(facecolor=COLOR_XGB, label="Señal emitida y retorno (sensor activo)"),
]
fig.legend(handles=leyenda_handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
           ncol=2, frameon=False, fontsize=7.6)
fig.tight_layout(rect=(0, 0.05, 1, 1))
guardar(fig, "fig_teledeteccion_pasivo_activo")

# ---------------------------------------------------------------------------
# 3. Humedad relativa: curva de presion de vapor de saturacion (Magnus) y
#    punto de rocio
# ---------------------------------------------------------------------------
a, b = 17.625, 243.04
T = np.linspace(-5, 30, 200)
es = 6.112 * np.exp((a * T) / (b + T))  # hPa, formula de Magnus-Tetens

fig, ax = plt.subplots(figsize=(6.6, 4.6))
linea_es, = ax.plot(T, es, color=COLOR_LSTM, linewidth=1.8, label="Presión de vapor de saturación $e_s(T)$")

T_ejemplo, Td_ejemplo = 18.0, 13.0
es_T = 6.112 * np.exp((a * T_ejemplo) / (b + T_ejemplo))
es_Td = 6.112 * np.exp((a * Td_ejemplo) / (b + Td_ejemplo))
hr_ejemplo = 100 * es_Td / es_T

punto_T = ax.scatter([T_ejemplo], [es_T], color=COLOR_OBS, zorder=5, s=32, label=f"Temperatura del aire, $T$ = {T_ejemplo:.0f} °C")
punto_Td = ax.scatter([Td_ejemplo], [es_Td], color=COLOR_XGB, zorder=5, s=32, label=f"Punto de rocío, $T_d$ = {Td_ejemplo:.0f} °C")
ax.plot([Td_ejemplo, Td_ejemplo], [0, es_Td], color=COLOR_XGB, linewidth=1.0, linestyle=":")
ax.plot([Td_ejemplo, T_ejemplo], [es_Td, es_Td], color=COLOR_GREEN_DARK, linewidth=1.3, linestyle="--")
ax.plot([T_ejemplo, T_ejemplo], [0, es_T], color=COLOR_OBS, linewidth=1.0, linestyle=":")

ax.annotate(f"$e_s(T)$ = {es_T:.1f} hPa\n(aire a esta temperatura)",
            xy=(T_ejemplo, es_T), xytext=(T_ejemplo + 1.0, es_T + 7.5),
            fontsize=7.2, color=COLOR_OBS, ha="left",
            arrowprops=dict(arrowstyle="-", color=COLOR_OBS, linewidth=0.6, shrinkA=2, shrinkB=2))
ax.annotate(f"$e_s(T_d)$ = {es_Td:.1f} hPa\n(aire saturado a esta $T$)",
            xy=(Td_ejemplo, es_Td), xytext=(-5.0, es_Td + 10.5),
            fontsize=7.2, color=COLOR_XGB, ha="left",
            arrowprops=dict(arrowstyle="-", color=COLOR_XGB, linewidth=0.6, shrinkA=2, shrinkB=2))
ax.text(T_ejemplo + 1.0, es_Td - 2.2, f"HR = 100 × $e_s(T_d)/e_s(T)$ = {hr_ejemplo:.0f}%",
        fontsize=7.6, color=COLOR_GREEN_DARK, fontweight="bold")

ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("Presión de vapor de saturación $e_s$ (hPa)")
ax.set_xlim(-5, 30)
ax.set_ylim(0, 48)
fig.legend(handles=[linea_es, punto_T, punto_Td], loc="upper center", bbox_to_anchor=(0.5, 0.03),
           ncol=1, frameon=False, fontsize=7.6)
fig.tight_layout(rect=(0, 0.13, 1, 1))
guardar(fig, "fig_humedad_relativa_punto_rocio")

# ---------------------------------------------------------------------------
# 4. Analisis de sensibilidad: barrido de un parametro de entrada (ejemplo
#    real, ventana temporal de esta tesis) -> efecto en la metrica de salida
# ---------------------------------------------------------------------------
try:
    df_sens = pd.read_csv(f"{DATA}/sensibilidad_ventana.csv")
    lstm = df_sens[df_sens["Modelo"] == "LSTM"].sort_values("Ventana")

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    linea_nse = ax.errorbar(lstm["Ventana"], lstm["NSE"], yerr=lstm["NSE_std"], marker="o",
                             color=COLOR_LSTM, linewidth=1.6, markersize=5, capsize=3,
                             label="NSE (LSTM), media ± desv. estándar entre 3 semillas de inicialización")
    ax.set_xlabel("Valor del parámetro barrido (ventana temporal $m$, días)")
    ax.set_ylabel("Métrica de salida del modelo (NSE)")
    ax.set_xticks(lstm["Ventana"])
    y_min = (lstm["NSE"] - lstm["NSE_std"]).min()
    y_max = (lstm["NSE"] + lstm["NSE_std"]).max()
    margen = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margen, y_max + margen)
    fig.legend(handles=[linea_nse], loc="upper center", bbox_to_anchor=(0.5, 0.17),
               ncol=1, frameon=False, fontsize=7.6)
    fig.text(0.5, 0.045,
             "El análisis de sensibilidad varía sistemáticamente una entrada del modelo (aquí, $m$) y mide\n"
             "el cambio resultante en una métrica de salida, para separar el efecto de esa entrada del\n"
             "ruido propio del entrenamiento (la barra de error).",
             fontsize=7.2, color=COLOR_MUTED, va="bottom", ha="center", style="italic")
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    guardar(fig, "fig_analisis_sensibilidad_concepto")
except FileNotFoundError:
    print("  (omitido) fig_analisis_sensibilidad_concepto: no se encontró sensibilidad_ventana.csv")

print("Tercera tanda de figuras conceptuales del marco teórico generada.")
