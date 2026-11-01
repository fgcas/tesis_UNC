#!/usr/bin/env python3
"""Genera la figura comparativa de la grilla ERA5-Land (0.1°, misma resolucion
nativa que PISCOp) sobre las 6 estaciones, en el mismo estilo visual que el
Mapa 2 (grilla PISCO) de generar_mapas.py, a partir del CSV ya extraido por
scripts/extraer_era5land_gee.py (no vuelve a llamar a la API de GEE)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

BASE = str(Path(__file__).resolve().parent.parent)
OUT = f"{BASE}/tesis/figuras"

COLOR_MUTED = "#52514e"
COLOR_ERA5 = "#2a78d6"   # mismo azul que COLOR_LSTM, para distinguir de PISCO (naranja)
COLOR_PISCO = "#eb6834"  # mismo naranja que COLOR_XGB, consistente con generar_mapas.py

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": COLOR_MUTED, "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a", "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ESTACIONES_COORDS = {
    'AUGUSTO_WEBERBAUER': (-78.49, -7.17, 2673),
    'ASUNCION': (-78.52, -7.33, 2270),
    'JESUS': (-78.39, -7.25, 2564),
    'LA_ENCANADA': (-78.33, -7.12, 2980),
    'NAMORA': (-78.33, -7.20, 2760),
    'SAN_JUAN': (-78.49, -7.30, 2253),
}
ESTACIONES_LABEL = {
    'AUGUSTO_WEBERBAUER': 'A. Weberbauer', 'ASUNCION': 'Asunción', 'JESUS': 'Jesús',
    'LA_ENCANADA': 'La Encañada', 'NAMORA': 'Namora', 'SAN_JUAN': 'San Juan'
}

# Geo-transformacion real de ERA5-Land (EPSG:4326), confirmada via la API:
# crs_transform = [0.1, 0, -180.05, 0, -0.1, 90.05] -> origen (-180.05, 90.05),
# resolucion 0.1 grados, por lo que el BORDE de cada pixel cae en multiplos
# exactos de 0.1 grados (el centro queda en el punto medio, x.x5).
RES_ERA5 = 0.1
ORIGEN_LON, ORIGEN_LAT = -180.05, 90.05


def fmt_lon(v):
    return f"{abs(v):.1f}°O"


def fmt_lat(v):
    return f"{abs(v):.1f}°S"


def generar_ticks(vmin, vmax, paso):
    n0 = round(vmin / paso)
    n1 = round(vmax / paso)
    return np.round(np.arange(n0, n1 + 1) * paso, 10)


def centro_pixel_era5(lon, lat):
    col = round((lon - ORIGEN_LON) / RES_ERA5 - 0.5)
    row = round((ORIGEN_LAT - lat) / RES_ERA5 - 0.5)
    centro_lon = ORIGEN_LON + (col + 0.5) * RES_ERA5
    centro_lat = ORIGEN_LAT - (row + 0.5) * RES_ERA5
    return centro_lon, centro_lat


def agregar_flecha_norte(ax, fig, ancho_in=0.34, alto_in=0.46, margen_in=0.07):
    from matplotlib.patches import FancyBboxPatch
    bbox = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_w_in, ax_h_in = bbox.width * fig_w_in, bbox.height * fig_h_in
    ancho, alto = ancho_in / ax_w_in, alto_in / ax_h_in
    margen_x, margen_y = margen_in / ax_w_in, margen_in / ax_h_in
    x0, y0 = 1 - margen_x - ancho, 1 - margen_y - alto
    cx = x0 + ancho / 2
    caja = FancyBboxPatch(
        (x0, y0), ancho, alto, transform=ax.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        facecolor="white", edgecolor=COLOR_MUTED, linewidth=0.7,
        alpha=0.92, zorder=9,
    )
    ax.add_patch(caja)
    ax.annotate("N", xy=(cx, y0 + alto * 0.24), xycoords="axes fraction", fontsize=9.5,
                fontweight="bold", ha="center", va="center", zorder=10)
    ax.annotate("", xy=(cx, y0 + alto * 0.93), xytext=(cx, y0 + alto * 0.42),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4), zorder=10)


def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    fig.savefig(f"{OUT}/{nombre}.png")
    plt.close(fig)
    print(f"  -> {nombre}.pdf / .png")


# ---------------------------------------------------------------------------
# Figura: grilla ERA5-Land y pixel asignado por estacion (estilo Mapa 2 PISCO)
# ---------------------------------------------------------------------------
print("Cargando ERA5-Land ya extraído...")
df = pd.read_csv(f"{BASE}/datos/era5land_estaciones.csv")
df["temp_C"] = df["temperature_2m"] - 273.15

# Temperatura media del primer año disponible, por estacion, como variable
# de color de la grilla (equivalente a "precip_media, año 1" del Mapa 2 PISCO)
primer_anio = df[pd.to_datetime(df["FECHA"]).dt.year == pd.to_datetime(df["FECHA"]).dt.year.min()]
temp_media_estacion = primer_anio.groupby("ESTACION")["temp_C"].mean()

margen = 0.02
lons = [v[0] for v in ESTACIONES_COORDS.values()]
lats = [v[1] for v in ESTACIONES_COORDS.values()]
xmin, xmax = min(lons) - margen, max(lons) + margen
ymin, ymax = min(lats) - margen, max(lats) + margen

# Grilla de pixeles ERA5-Land que cubre el area de interes (recorte ancho,
# igual criterio que el Mapa 2 PISCO: +/-0.15 grados de margen). Los bordes
# de cada celda caen en multiplos exactos de RES_ERA5 desplazados desde
# ORIGEN_LON/ORIGEN_LAT (confirmado via crs_transform de la API de GEE).
def bordes_alineados(vmin, vmax, origen, paso):
    n0 = np.floor((vmin - origen) / paso)
    n1 = np.ceil((vmax - origen) / paso)
    return origen + np.arange(n0, n1 + 1) * paso

lon_bordes = bordes_alineados(xmin - 0.15, xmax + 0.15, ORIGEN_LON, RES_ERA5)
lat_bordes = np.sort(bordes_alineados(ymin - 0.15, ymax + 0.15, ORIGEN_LAT, -RES_ERA5))

fig, ax = plt.subplots(figsize=(6.6, 6.0))

# Dibuja cada celda de la grilla como un rectangulo (sin malla continua de
# datos reales, ya que solo se extrajeron los 6 puntos puntuales, no el
# raster completo); dentro de cada celda que contiene una estacion se anota
# su temperatura media del año 1, a modo de muestra representativa.
for lon0 in lon_bordes[:-1]:
    for lat0 in lat_bordes[:-1]:
        ax.add_patch(plt.Rectangle((lon0, lat0), RES_ERA5, RES_ERA5,
                                    fill=True, facecolor="#eef3fa",
                                    edgecolor=COLOR_MUTED, linewidth=0.3, zorder=1))

# Las estaciones que caen en el MISMO pixel ERA5-Land se agrupan para
# pintar esa celda una sola vez con un color propio (en vez de superponer
# rectangulos semitransparentes del mismo color, que a simple vista se
# funden con la celda vecina si son contiguas, como ocurria con La
# Encañada/Namora en la primera version de esta figura).
pixeles = {}
for est, (lon, lat, alt) in ESTACIONES_COORDS.items():
    px = centro_pixel_era5(lon, lat)
    pixeles.setdefault(px, []).append(est)

PALETA_GRUPOS = ["#2a78d6", "#eb6834", "#2ca02c", "#9467bd", "#d62728"]
for i, ((px_lon, px_lat), ests) in enumerate(pixeles.items()):
    color = PALETA_GRUPOS[i % len(PALETA_GRUPOS)]
    ax.add_patch(plt.Rectangle((px_lon - RES_ERA5/2, px_lat - RES_ERA5/2), RES_ERA5, RES_ERA5,
                                fill=True, facecolor=color, alpha=0.30,
                                edgecolor=color, linewidth=2.2, zorder=3))
    if len(ests) > 1:
        print(f"  Comparten pixel ERA5-Land en ({px_lon:.2f}, {px_lat:.2f}): {ests}")

for est, (lon, lat, alt) in ESTACIONES_COORDS.items():
    ax.scatter(lon, lat, s=45, color="black", marker="o", zorder=6, edgecolor="white", linewidth=0.8)
    etiqueta = f"{ESTACIONES_LABEL[est]}\n{temp_media_estacion[est]:.1f} °C"
    ax.annotate(etiqueta, (lon, lat), fontsize=7.0, xytext=(5, 4),
                textcoords="offset points", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

ax.set_xlim(xmin - 0.05, xmax + 0.05)
ax.set_ylim(ymin - 0.05, ymax + 0.05)

paso = 0.1
xlim, ylim = ax.get_xlim(), ax.get_ylim()
lon_ticks = generar_ticks(xlim[0], xlim[1], paso)
lat_ticks = generar_ticks(ylim[0], ylim[1], paso)
ax.set_xticks(lon_ticks)
ax.set_xticklabels([fmt_lon(v) for v in lon_ticks], fontsize=7.3)
ax.set_yticks(lat_ticks)
ax.set_yticklabels([fmt_lat(v) for v in lat_ticks], fontsize=7.3)
ax.grid(True, color="white", linewidth=0.5, alpha=0.5, linestyle="-", zorder=2)

for spine in ax.spines.values():
    spine.set_visible(True); spine.set_edgecolor(COLOR_MUTED); spine.set_linewidth(0.8)

ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_title("Grilla de ERA5-Land (0.1°) y píxel de vecino más cercano por estación",
             loc="left", fontweight="bold", fontsize=10)

leyenda = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markeredgecolor="white", markersize=7, label="Estación SENAMHI"),
    Line2D([0], [0], color=COLOR_MUTED, linewidth=2.2, label="Píxel ERA5-Land (0.1°); mismo color = mismo píxel"),
]
ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
          frameon=False, fontsize=7.5)
fig.tight_layout()
agregar_flecha_norte(ax, fig)
guardar(fig, "mapa_grilla_era5land")

print(f"Resolución de la grilla ERA5-Land: {RES_ERA5}° lon x {RES_ERA5}° lat (misma que PISCOp)")

# ---------------------------------------------------------------------------
# Figura comparativa: series de ejemplo PISCO vs ERA5-Land (temperatura no
# existe en PISCOp, asi que se compara contra la propia precipitacion
# observada para mostrar que la escala temporal es la misma, 2000-2025) y
# un pequeño panel resumen de variables nuevas disponibles por estacion.
# ---------------------------------------------------------------------------
print("\nGenerando panel comparativo de variables ERA5-Land por estación...")
df["FECHA"] = pd.to_datetime(df["FECHA"])
resumen = df.groupby("ESTACION").agg(
    temp_media_C=("temp_C", "mean"),
    temp_min_C=("temperature_2m_min", lambda s: (s - 273.15).mean()),
    temp_max_C=("temperature_2m_max", lambda s: (s - 273.15).mean()),
    presion_hPa=("surface_pressure", lambda s: (s / 100).mean()),
    viento_u_ms=("u_component_of_wind_10m", "mean"),
    viento_v_ms=("v_component_of_wind_10m", "mean"),
).reindex(list(ESTACIONES_COORDS.keys()))
resumen.index = [ESTACIONES_LABEL[e] for e in resumen.index]

fig2, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

axes[0].barh(resumen.index, resumen["temp_media_C"], color=COLOR_ERA5, alpha=0.85,
             label="Media")
axes[0].scatter(resumen["temp_min_C"], resumen.index, color="#1a1a1a", marker="|", s=120,
                 label="Mín. media")
axes[0].scatter(resumen["temp_max_C"], resumen.index, color=COLOR_PISCO, marker="|", s=120,
                 label="Máx. media")
axes[0].set_xlabel("Temperatura 2 m (°C)")
axes[0].set_title("(a) Temperatura media 2000–2025", loc="left", fontweight="bold", fontsize=9)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")

viento_mag = np.sqrt(resumen["viento_u_ms"]**2 + resumen["viento_v_ms"]**2)
axes[1].barh(resumen.index, resumen["presion_hPa"], color=COLOR_MUTED, alpha=0.75)
ax2b = axes[1].twiny()
ax2b.scatter(viento_mag, resumen.index, color=COLOR_ERA5, marker="D", s=40, zorder=5,
             label="Viento (m/s)")
axes[1].set_xlabel("Presión superficial media (hPa)")
ax2b.set_xlabel("Magnitud del viento medio (m/s)", color=COLOR_ERA5)
ax2b.tick_params(axis="x", colors=COLOR_ERA5)
axes[1].set_xlim(resumen["presion_hPa"].min() - 5, resumen["presion_hPa"].max() + 5)
axes[1].set_title("(b) Presión y viento medios 2000–2025", loc="left", fontweight="bold", fontsize=9)

fig2.tight_layout()
guardar(fig2, "comparacion_variables_era5land")

print("\nResumen de variables ERA5-Land por estación:")
print(resumen.round(2))
