#!/usr/bin/env python3
"""
Genera el diagrama de flujo metodológico completo: desde los datos crudos
(SENAMHI + PISCO) hasta las métricas finales, siguiendo exactamente las
7 fases descritas en el Capítulo III (Técnicas de análisis e interpretación
de datos) y las secciones del notebook 001_Tesis_FGCQ.ipynb. Mismo estilo que el
resto de figuras del documento (generar_figuras_marco_teorico.py).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches

OUT = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_GREEN = "#1baf7a"
COLOR_OBS = "#1a1a1a"
COLOR_GRID = "#d8d8d4"
COLOR_MUTED = "#52514e"
COLOR_NEUTRO = "#6b6b68"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "text.color": "#1a1a1a",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    fig.savefig(f"{OUT}/{nombre}.png")
    plt.close(fig)
    print(f"  -> {nombre}.pdf / .png")


def alpha_fondo(color, factor=0.12):
    """Devuelve el color mezclado con blanco para usarlo como fondo suave de caja."""
    r, g, b = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    r = int(r + (255 - r) * (1 - factor))
    g = int(g + (255 - g) * (1 - factor))
    b = int(b + (255 - b) * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Definición del contenido de cada bloque del flujo
# ---------------------------------------------------------------------------
bloques = [
    dict(
        titulo="Datos crudos",
        lineas=["SENAMHI: 6 estaciones, registro diario 2000–2025",
                "PISCOp v03: producto grillado (NetCDF, 0.1°)"],
        color=COLOR_NEUTRO, grupo=None,
    ),
    dict(
        titulo="Fase 1 — Limpieza y análisis exploratorio",
        lineas=["Identificación y reemplazo de códigos no numéricos (S/D, trazas)",
                "EDA: estacionalidad, doble masa, mapa de calor, cadenas de Markov"],
        color=COLOR_LSTM, grupo="A",
    ),
    dict(
        titulo="Fase 2 — Imputación multivariada sin fuga",
        lineas=["Partición cronológica (70% train) antes de ajustar cualquier estadístico",
                "MICE (ExtraTrees) ajustado solo en train → transform de la serie completa"],
        color=COLOR_LSTM, grupo="A",
    ),
    dict(
        titulo="Fase 3 — Integración y escalado",
        lineas=["Extracción PISCO (píxel más cercano) + forzamiento estacional (sen/cos)",
                "RobustScaler y MinMaxScaler ajustados solo en train → transform completo"],
        color=COLOR_LSTM, grupo="A",
    ),
    dict(
        titulo="Fase 4 — Reconstrucción tensorial y partición",
        lineas=["Ventana de retraso de 15 días (teorema de Takens)",
                "Partición cronológica 70/15/15 (entrenamiento / validación / prueba)"],
        color=COLOR_LSTM, grupo="A",
    ),
    dict(
        titulo="Fase 5 — Búsqueda de hiperparámetros",
        lineas=["XGBoost: RandomizedSearchCV + TimeSeriesSplit (k=5)",
                "LSTM: grid reducido, validación simple (limitación de cómputo, 8 GB RAM)"],
        color=COLOR_XGB, grupo="B",
    ),
    dict(
        titulo="Fase 6 — Entrenamiento final",
        lineas=["LSTM: función de pérdida asimétrica (mse_asimetrico)",
                "XGBoost: objetivo asimétrico personalizado (gradiente / hessiano equivalente)"],
        color=COLOR_XGB, grupo="B",
    ),
    dict(
        titulo="Fase 7 — Evaluación estadística",
        lineas=["RMSE, NSE y KGE — conjunto de prueba cronológico (6 estaciones)",
                "Escenarios de vacíos simulados A/B/C (10% / 20% / 30%)"],
        color=COLOR_GREEN, grupo="C",
    ),
    dict(
        titulo="Resultado final",
        lineas=["Tabla comparativa LSTM vs. XGBoost por estación y escenario",
                "Contraste con la hipótesis general de la investigación"],
        color=COLOR_NEUTRO, grupo=None,
    ),
]

grupos_meta = {
    "A": ("PREPARACIÓN DE DATOS", COLOR_LSTM),
    "B": ("MODELADO", COLOR_XGB),
    "C": ("EVALUACIÓN", COLOR_GREEN),
}

n = len(bloques)
alto_caja = 1.0
espacio = 0.42
alto_total = n * alto_caja + (n - 1) * espacio + 1.0

fig, ax = plt.subplots(figsize=(7.3, alto_total * 0.62))
ax.set_xlim(0, 10)
ax.set_ylim(0, alto_total)
ax.axis("off")

y = alto_total - 0.6
centros_y = []
for b in bloques:
    centros_y.append(y - alto_caja / 2)
    y -= (alto_caja + espacio)

# --- Franjas de agrupación (fondo) detrás de las fases 1-4 / 5-6 / 7 ---
for clave, (etiqueta, color) in grupos_meta.items():
    idxs = [i for i, b in enumerate(bloques) if b["grupo"] == clave]
    y_top = centros_y[idxs[0]] + alto_caja / 2 + espacio / 2
    y_bot = centros_y[idxs[-1]] - alto_caja / 2 - espacio / 2
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.15, y_bot), 0.55, y_top - y_bot,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=alpha_fondo(color, 0.18), zorder=0,
    ))
    ax.text(0.42, (y_top + y_bot) / 2, etiqueta, rotation=90, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=color, zorder=1)

# --- Cajas de cada bloque ---
x0, ancho = 1.05, 8.4
for i, (b, cy) in enumerate(zip(bloques, centros_y)):
    es_extremo = b["grupo"] is None
    fill = alpha_fondo(b["color"], 0.10 if not es_extremo else 0.06)
    borde = b["color"]
    box = FancyBboxPatch(
        (x0, cy - alto_caja / 2), ancho, alto_caja,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor=borde, facecolor=fill, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x0 + 0.22, cy + alto_caja / 2 - 0.24, b["titulo"],
            fontsize=9.3, fontweight="bold", color="#1a1a1a", va="top", zorder=3)
    texto_lineas = "\n".join(f"• {l}" for l in b["lineas"])
    ax.text(x0 + 0.22, cy + alto_caja / 2 - 0.50, texto_lineas,
            fontsize=7.6, color=COLOR_MUTED, va="top", zorder=3, linespacing=1.5)

    if i < n - 1:
        y_ini = cy - alto_caja / 2
        y_fin = centros_y[i + 1] + alto_caja / 2
        ax.add_patch(FancyArrowPatch(
            (x0 + ancho / 2, y_ini), (x0 + ancho / 2, y_fin),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
            color=COLOR_MUTED, zorder=1, shrinkA=0, shrinkB=0,
        ))

fig.tight_layout(pad=0.3)
guardar(fig, "fig_flujo_metodologico")
print("Diagrama de flujo metodológico generado.")
