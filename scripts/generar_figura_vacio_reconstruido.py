#!/usr/bin/env python3
"""
Genera la figura y la tabla que comparan, para un tramo contiguo de fechas
reales del conjunto de prueba, la precipitación observada frente a la
reconstruida por LSTM y XGBoost, resaltando un sub-tramo tratado como vacío
simulado (oculto a las métricas, con su valor real sí conocido).

No es un vacío real de SENAMHI: los vacíos reales más largos del registro
(ver TABLA II) caen en su totalidad fuera del conjunto de prueba, por lo que
comparar un modelo contra esos tramos equivaldría a comparar contra la
imputación MICE y no contra una medición de campo. Este tramo, en cambio,
tiene observación real conocida para las tres series.

Lee tesis/figuras_data/test_series_fisicas_18var.npz (ya existente, sin
recomputar nada) y escribe PDF+PNG en tesis/figuras/ y un CSV resumen en
tesis/figuras_data/.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = str(Path(__file__).resolve().parent.parent / "tesis")
DATA = f"{BASE}/figuras_data"
OUT = f"{BASE}/figuras"
os.makedirs(OUT, exist_ok=True)

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_OBS = "#1a1a1a"
COLOR_GRID = "#d8d8d4"
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#52514e"
COLOR_VACIO = "#f4d35e"

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

def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    fig.savefig(f"{OUT}/{nombre}.png")
    plt.close(fig)
    print(f"  -> {nombre}.pdf / .png")

# ---------------------------------------------------------------------------
# Carga de datos (ya calculados, sin reentrenar)
# ---------------------------------------------------------------------------
series = np.load(f"{DATA}/test_series_fisicas_18var.npz", allow_pickle=True)
obs_all, pred_lstm_all, pred_xgb_all = series["obs"], series["pred_lstm"], series["pred_xgb"]
idx_est = 0  # Augusto Weberbauer, misma estación que fig_hietograma_excedencia

# El test set son los últimos 1423 días cronológicos de la serie 2000-2025.
fechas_test = pd.date_range(end="2025-12-31", periods=obs_all.shape[0], freq="D")

# Mismo tramo de 100 días que ya usa fig_hietograma_excedencia (índice
# 150-250 del test set, jul-oct 2022), para mantener coherencia visual con
# esa figura. Sub-tramo central resaltado como "vacío simulado" (15 días,
# con margen de contexto a ambos lados).
i0, win = 150, 100
j0, j1 = 70, 70 + 15  # offset dentro de la ventana

fechas_ventana = fechas_test[i0:i0 + win]
obs = obs_all[i0:i0 + win, idx_est]
lstm = pred_lstm_all[i0:i0 + win, idx_est]
xgb = pred_xgb_all[i0:i0 + win, idx_est]

fecha_ini_vacio = fechas_ventana[j0]
fecha_fin_vacio = fechas_ventana[j1 - 1]

# ---------------------------------------------------------------------------
# Tabla resumen: observado vs. reconstruido en el tramo oculto
# ---------------------------------------------------------------------------
print("Tabla: resumen del tramo oculto reconstruido")
obs_vacio = obs[j0:j1]
lstm_vacio = lstm[j0:j1]
xgb_vacio = xgb[j0:j1]

def metricas(obs_, pred_):
    error = pred_ - obs_
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    sesgo = np.mean(error)
    r = np.corrcoef(obs_, pred_)[0, 1]
    return rmse, mae, sesgo, r

rmse_lstm, mae_lstm, sesgo_lstm, r_lstm = metricas(obs_vacio, lstm_vacio)
rmse_xgb, mae_xgb, sesgo_xgb, r_xgb = metricas(obs_vacio, xgb_vacio)

resumen = pd.DataFrame({
    "fecha": fechas_ventana[j0:j1].strftime("%Y-%m-%d"),
    "observado_mm": obs_vacio.round(1),
    "lstm_mm": lstm_vacio.round(1),
    "xgboost_mm": xgb_vacio.round(1),
    "error_abs_lstm": np.abs(obs_vacio - lstm_vacio).round(1),
    "error_abs_xgb": np.abs(obs_vacio - xgb_vacio).round(1),
})
resumen.to_csv(f"{DATA}/vacio_reconstruido_detalle.csv", index=False)
print(resumen.to_string(index=False))
print(f"\nLSTM    -> RMSE={rmse_lstm:.3f}  MAE={mae_lstm:.3f}  sesgo={sesgo_lstm:+.3f}  r={r_lstm:.3f}")
print(f"XGBoost -> RMSE={rmse_xgb:.3f}  MAE={mae_xgb:.3f}  sesgo={sesgo_xgb:+.3f}  r={r_xgb:.3f}")

pd.DataFrame([{
    "estacion": "AUGUSTO_WEBERBAUER",
    "fecha_inicio_vacio": fecha_ini_vacio.strftime("%Y-%m-%d"),
    "fecha_fin_vacio": fecha_fin_vacio.strftime("%Y-%m-%d"),
    "dias_vacio": j1 - j0,
    "rmse_lstm": round(rmse_lstm, 3), "mae_lstm": round(mae_lstm, 3),
    "sesgo_lstm": round(sesgo_lstm, 3), "r_lstm": round(r_lstm, 3),
    "rmse_xgb": round(rmse_xgb, 3), "mae_xgb": round(mae_xgb, 3),
    "sesgo_xgb": round(sesgo_xgb, 3), "r_xgb": round(r_xgb, 3),
}]).to_csv(f"{DATA}/vacio_reconstruido_resumen.csv", index=False)

# ---------------------------------------------------------------------------
# Figura, panel (a): mismo tramo y estilo que fig_hietograma_excedencia
# (línea simple, sin marcador), con fechas reales y el tramo oculto resaltado.
# Panel (b): detalle del tramo oculto en gráfico de agujas (stem), un
# marcador distinto por serie, ligeramente desplazadas en X para no solaparse.
# ---------------------------------------------------------------------------
print("\nFigura: vacío simulado reconstruido (contexto + detalle en agujas)")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.6),
                                gridspec_kw={"height_ratios": [1.1, 1]})

ax1.axvspan(fecha_ini_vacio, fecha_fin_vacio, color=COLOR_VACIO, alpha=0.35,
            label="Tramo tratado como vacío (oculto a las métricas)", zorder=0)
ax1.plot(fechas_ventana, obs, color=COLOR_OBS, linewidth=1.3, label="Observado (SENAMHI)")
ax1.plot(fechas_ventana, lstm, color=COLOR_LSTM, linewidth=1.1, label="Reconstrucción LSTM")
ax1.plot(fechas_ventana, xgb, color=COLOR_XGB, linewidth=1.1, linestyle="--", label="Reconstrucción XGBoost")
ax1.set_ylabel("Precipitación (mm/día)")
ax1.set_title("(a) Contexto del tramo de 100 días, con el vacío simulado resaltado",
               loc="left", fontweight="bold", fontsize=9)
ax1.tick_params(axis="x", rotation=30)
for lbl in ax1.get_xticklabels():
    lbl.set_ha("right")
ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=7.5)

# Panel (b): gráfico de agujas (stem) del tramo oculto, un marcador por serie
etiquetas = [f.strftime("%d-%b") for f in fechas_ventana[j0:j1]]
x = np.arange(len(etiquetas))
despl = 0.12  # desplazamiento horizontal entre series para que no se solapen

m1 = ax2.stem(x - despl, obs_vacio, linefmt=COLOR_OBS, markerfmt="o", basefmt=" ")
plt.setp(m1[0], markerfacecolor=COLOR_OBS, markeredgecolor=COLOR_OBS, markersize=6, zorder=3)
plt.setp(m1[1], linewidth=1.3, color=COLOR_OBS)

m2 = ax2.stem(x, lstm_vacio, linefmt=COLOR_LSTM, markerfmt="s", basefmt=" ")
plt.setp(m2[0], markerfacecolor=COLOR_LSTM, markeredgecolor=COLOR_LSTM, markersize=5.5, zorder=3)
plt.setp(m2[1], linewidth=1.3, color=COLOR_LSTM)

m3 = ax2.stem(x + despl, xgb_vacio, linefmt=COLOR_XGB, markerfmt="^", basefmt=" ")
plt.setp(m3[0], markerfacecolor=COLOR_XGB, markeredgecolor=COLOR_XGB, markersize=5.5, zorder=3)
plt.setp(m3[1], linewidth=1.3, color=COLOR_XGB)

ax2.set_xticks(x)
ax2.set_xticklabels(etiquetas, rotation=45, ha="right")
ax2.set_ylabel("Precipitación (mm/día)")
ax2.set_xlabel("Fecha (tramo oculto)")
ax2.set_title("(b) Detalle del tramo oculto: observado vs. reconstruido, por día",
               loc="left", fontweight="bold", fontsize=9)
ax2.legend([m1[0], m2[0], m3[0]],
           ["Observado (SENAMHI)", "Reconstrucción LSTM", "Reconstrucción XGBoost"],
           frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=7.5)

fig.tight_layout()
guardar(fig, "fig_vacio_reconstruido")

print("\nListo.")
