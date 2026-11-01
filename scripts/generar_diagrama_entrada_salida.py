#!/usr/bin/env python3
"""Diagrama explicativo (version visual, sin jerga tecnica de hiperparametros):
como entra la data a los modelos LSTM y XGBoost, como cada uno "aprende" en un
ciclo que se repite (observar -> predecir -> comparar -> corregir), y que
salida final producen. Pensado para que un jurado sin formacion en machine
learning entienda el mecanismo sin necesidad de conocer terminos como Dropout,
epocas, Adam o gradiente/hessiano -- esos detalles quedan solo en el texto de
la metodologia, no en la figura. Complementa fig_flujo_metodologico.png (que
cubre las 7 fases del pipeline completo, incluida la limpieza/imputacion).
Metafora: un embudo por modelo (los datos entran anchos por arriba) cuyo
"cuello" es un ciclo tipo reloj que se repite muchas veces, angostandose hasta
una unica gota de salida (la prediccion)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = "/Users/frankcasanova/Documents/TESIS UNC/CODIGO_TESIS/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_GREEN = "#1baf7a"
COLOR_MUTED = "#52514e"
COLOR_NEUTRO = "#6b6b68"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "text.color": "#1a1a1a",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    fig.savefig(f"{OUT}/{nombre}.png")
    plt.close(fig)
    print(f"  -> {nombre}.pdf / .png")


def alpha_fondo(color, factor=0.10):
    r, g, b = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    r = int(r + (255 - r) * (1 - factor))
    g = int(g + (255 - g) * (1 - factor))
    b = int(b + (255 - b) * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def caja(ax, x0, y0, ancho, alto, color, titulo, lineas, fs_titulo=11.4, fs_linea=9.3):
    box = FancyBboxPatch(
        (x0, y0), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, edgecolor=color, facecolor=alpha_fondo(color, 0.08), zorder=2,
    )
    ax.add_patch(box)
    ax.text(x0 + 0.25, y0 + alto - 0.32, titulo, fontsize=fs_titulo,
             fontweight="bold", color="#000000", va="top", zorder=3)
    ax.text(x0 + 0.25, y0 + alto - 0.70, "\n".join(lineas), fontsize=fs_linea,
             color="#000000", va="top", zorder=3, linespacing=1.7)
    return box


def embudo(ax, cx, color, titulo, descripcion):
    """Dibuja un embudo (boca ancha -> cuello constante -> pico) con un ciclo
    de 4 pasos tipo reloj en el cuello, y devuelve el punto de salida (gota)."""
    top_hw, neck_hw = 2.6, 1.8
    y_top, y_neck_top, y_neck_bot, y_point = 10.6, 9.5, 6.1, 5.1

    verts = [
        (cx - top_hw, y_top), (cx + top_hw, y_top),
        (cx + neck_hw, y_neck_top), (cx + neck_hw, y_neck_bot),
        (cx, y_point),
        (cx - neck_hw, y_neck_bot), (cx - neck_hw, y_neck_top),
    ]
    ax.add_patch(Polygon(verts, closed=True, linewidth=1.6, edgecolor=color,
                          facecolor=alpha_fondo(color, 0.07), zorder=2))

    # Título y descripción en lenguaje simple, dentro de la boca del embudo
    ax.text(cx, y_top - 0.32, titulo, fontsize=11.5, fontweight="bold",
             color="#000000", ha="center", va="top", zorder=3)
    ax.text(cx, y_top - 0.68, descripcion, fontsize=8.4, color="#000000",
             ha="center", va="top", style="italic", zorder=3)

    # --- Ciclo de 4 pasos (tipo reloj) dentro del cuello del embudo --------
    cy_mid = (y_neck_top + y_neck_bot) / 2
    r_v, r_h = 1.25, 1.3
    p_top = (cx, cy_mid + r_v)
    p_der = (cx + r_h, cy_mid)
    p_bot = (cx, cy_mid - r_v)
    p_izq = (cx - r_h, cy_mid)

    for (pa, pb, rad) in [(p_top, p_der, -0.35), (p_der, p_bot, -0.35),
                           (p_bot, p_izq, -0.35), (p_izq, p_top, -0.35)]:
        ax.add_patch(FancyArrowPatch(pa, pb, connectionstyle=f"arc3,rad={rad}",
                     arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                     color=color, alpha=0.85, zorder=3,
                     shrinkA=18, shrinkB=18))

    ax.text(*p_top, "1. Observa\nel pasado", fontsize=8.6, ha="center",
             va="center", color="#000000", fontweight="bold", zorder=4,
             linespacing=1.3)
    ax.text(*p_der, "2. Predice", fontsize=8.6, ha="center", va="center",
             color="#000000", fontweight="bold", zorder=4)
    ax.text(*p_bot, "3. Compara\ncon lo real", fontsize=8.6,
             ha="center", va="center", color="#000000", fontweight="bold",
             zorder=4, linespacing=1.3)
    ax.text(*p_izq, "4. Corrige", fontsize=8.6, ha="center", va="center",
             color="#000000", fontweight="bold", zorder=4)

    ax.text(cx, y_neck_bot - 0.06, "se repite cientos de veces",
             fontsize=7.6, style="italic", color="#000000", ha="center",
             va="center", zorder=3)

    # --- Salida: una gota con la predicción ---------------------------------
    ax.scatter([cx], [y_point], s=70, color=color, zorder=4,
               edgecolor="white", linewidth=0.8)
    ax.text(cx, y_point - 0.35, "Predicción de lluvia\npara el día siguiente",
             fontsize=8.4, ha="center", va="top", color="#000000", zorder=3,
             linespacing=1.3, fontweight="bold")
    return (cx, y_point - 0.75)


fig, ax = plt.subplots(figsize=(11.2, 9.0))
ax.set_xlim(0, 14)
ax.set_ylim(2.0, 13.2)
ax.axis("off")

# --- Entrada compartida ------------------------------------------------------
caja(ax, 0.4, 10.9, 13.2, 2.0, COLOR_NEUTRO,
     "Lo que cada modelo observa antes de predecir (últimos 15 días)",
     ["• Lluvia registrada en las 6 estaciones de SENAMHI",
      "• Estimación satelital de lluvia de esos mismos días (PISCO)",
      "• La época del año, para saber si es temporada lluviosa o seca"],
     fs_titulo=11.8, fs_linea=9.5)

ax.add_patch(FancyArrowPatch((4.2, 10.9), (3.6, 10.65), arrowstyle="-|>",
             mutation_scale=13, linewidth=1.3, color=COLOR_LSTM, zorder=1))
ax.add_patch(FancyArrowPatch((9.8, 10.9), (10.4, 10.65), arrowstyle="-|>",
             mutation_scale=13, linewidth=1.3, color=COLOR_XGB, zorder=1))

# --- Dos embudos: uno por modelo ---------------------------------------------
salida_lstm = embudo(ax, 3.6, COLOR_LSTM, "Modelo 1: red LSTM",
                      "ve los 15 días uno tras otro, como una memoria")
salida_xgb = embudo(ax, 10.4, COLOR_XGB, "Modelo 2: árboles XGBoost",
                     "combina árboles simples que se corrigen entre sí")

# --- Convergencia hacia el resultado final -----------------------------------
ax.add_patch(FancyArrowPatch(salida_lstm, (6.3, 4.1), arrowstyle="-|>",
             mutation_scale=13, linewidth=1.3, color=COLOR_LSTM, zorder=1,
             connectionstyle="arc3,rad=0.15"))
ax.add_patch(FancyArrowPatch(salida_xgb, (7.7, 4.1), arrowstyle="-|>",
             mutation_scale=13, linewidth=1.3, color=COLOR_XGB, zorder=1,
             connectionstyle="arc3,rad=-0.15"))

caja(ax, 2.4, 2.3, 9.2, 1.8, COLOR_GREEN,
     "Resultado final",
     ["Se compara la predicción de cada modelo con la lluvia que",
      "realmente cayó (con indicadores estadísticos como RMSE, NSE",
      "y KGE) para ver cuál de los dos acierta mejor."],
     fs_titulo=11.5, fs_linea=9.3)

fig.tight_layout(pad=0.3)
guardar(fig, "fig_entrada_salida_modelos")
print("Diagrama entrada-modelo-salida (version visual/embudo) generado.")
