#!/usr/bin/env python3
"""Genera mapas georreferenciados: (1) ubicacion de las 6 estaciones sobre
basemap real (contextily/OSM) con inset de Peru, (2) grilla PISCO real con
los pixeles de vecino mas cercano seleccionados para cada estacion."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import contextily as cx
import xarray as xr

BASE = str(Path(__file__).resolve().parent.parent)
OUT = f"{BASE}/tesis/figuras"

COLOR_LSTM = "#2a78d6"
COLOR_XGB = "#eb6834"
COLOR_MUTED = "#52514e"

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

def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    fig.savefig(f"{OUT}/{nombre}.png")
    plt.close(fig)
    print(f"  -> {nombre}.pdf / .png")

def agregar_flecha_norte(ax, fig, ancho_in=0.34, alto_in=0.46, margen_in=0.07):
    """Rosa de los vientos (N + flecha) con caja de fondo blanca semitransparente,
    para que sea legible sin importar lo que haya debajo (tile OSM, grilla de
    color, etc.). La letra "N" va claramente separada del trazo de la flecha
    (antes quedaba dentro del propio trazo, superpuesta e ilegible). El tamano
    se fija en pulgadas (no en fraccion de los ejes) para que la caja quede
    compacta sin importar el tamano del mapa."""
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

# ---------------------------------------------------------------------------
# MAPA 1: Ubicacion de las estaciones (basemap real OSM) + inset Peru
# ---------------------------------------------------------------------------
lons = [v[0] for v in ESTACIONES_COORDS.values()]
lats = [v[1] for v in ESTACIONES_COORDS.values()]
margen = 0.02
xmin, xmax = min(lons) - margen, max(lons) + margen
ymin, ymax = min(lats) - margen, max(lats) + margen

# El Mapa 1 usa una extension geografica fija, delimitada a mano (a pedido
# del asesor) en vez de derivarla de "margen": longitud 78.7-78.2 O, latitud
# 7.4-7.0 S. Cubre holgadamente las 6 estaciones y dejo margen para que
# ninguna etiqueta se salga del recuadro.
xmin1, xmax1 = -78.7, -78.2
ymin1, ymax1 = -7.4, -7.0

# Posicion de ejes fijada explicitamente en pulgadas (en vez de dejar que
# tight_layout()/constrained_layout() la infieran). El aspecto fijo (equal)
# que impone imshow sobre coordenadas geograficas hace que, si el recuadro de
# ejes que matplotlib reserva automaticamente para el mapa no tiene ya
# exactamente el aspecto de los datos, la imagen se encoja dentro de ese
# recuadro dejando una franja en blanco (tipicamente a la derecha). Al fijar
# el recuadro con el aspecto correcto desde el inicio, el mapa lo llena por
# completo sin recorte interno.
import pyproj
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

img, ext = cx.bounds2img(
    xmin1, ymin1, xmax1, ymax1, ll=True, source=cx.providers.OpenStreetMap.Mapnik, zoom=11,
    headers={"User-Agent": "TesisUNC-IngenieriaHidraulica-CajamarcaLSTM-XGBoost/1.0 "
                            "(uso academico, contacto: frankcasanova50@gmail.com)"},
)
# Los tiles vienen alineados a su propia cuadricula y cubren MAS que lo pedido.
# Los limites del eje se fijan a la extension pedida (en Web Mercator) y la caja
# del mapa se dimensiona con esa misma proporcion, asi el mapa llena el recuadro
# sin franjas en blanco y el marco negro queda pegado a los numeros.
lim_x0, lim_y0 = transformer.transform(xmin1, ymin1)
lim_x1, lim_y1 = transformer.transform(xmax1, ymax1)
aspecto = (lim_x1 - lim_x0) / (lim_y1 - lim_y0)
ancho_total_in = 7.0
margen_izq_in, margen_der_in = 0.55, 0.55
margen_sup_in, margen_inf_in = 0.35, 0.35

ancho_mapa_in = ancho_total_in - margen_izq_in - margen_der_in
alto_mapa_in = ancho_mapa_in / aspecto
alto_total_in = alto_mapa_in + margen_sup_in + margen_inf_in

fig = plt.figure(figsize=(ancho_total_in, alto_total_in))
ax = fig.add_axes([
    margen_izq_in / ancho_total_in,
    margen_inf_in / alto_total_in,
    ancho_mapa_in / ancho_total_in,
    alto_mapa_in / alto_total_in,
])
ax.imshow(img, extent=ext, interpolation="bilinear")

OFFSETS = {
    'AUGUSTO_WEBERBAUER': (7, 7), 'ASUNCION': (7, 0), 'JESUS': (7, 7),
    'LA_ENCANADA': (7, 7), 'NAMORA': (7, 7), 'SAN_JUAN': (9, 2),
}
for est, (lon, lat, alt) in ESTACIONES_COORDS.items():
    x, y = transformer.transform(lon, lat)
    ax.scatter(x, y, s=45, color="black", edgecolor="white", linewidth=1.3, zorder=5)
    txt = ax.annotate(f"{ESTACIONES_LABEL[est]}\n{alt} msnm", (x, y), fontsize=7.8,
                       xytext=OFFSETS[est], textcoords="offset points", color="#1a1a1a",
                       fontweight="bold", zorder=6)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

# Grilla de coordenadas (graticula) para mejorar la georreferenciacion visual:
# lineas cada 2 km en UTM zona 17S (EPSG:32717, WGS84), con etiquetas
# Este/Norte en los bordes. El eje del mapa esta en Web Mercator (lo que
# exige contextily para el basemap); un segundo transformer convierte
# UTM -> lon/lat, que luego se lleva a Web Mercator para posicionar cada
# linea de grilla, pero el texto de la etiqueta usa el valor UTM original.
to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32717", always_xy=True)
from_utm = pyproj.Transformer.from_crs("EPSG:32717", "EPSG:4326", always_xy=True)

def fmt_este(v):
    return f"{v:,.0f} E".replace(",", " ")

def fmt_norte(v):
    return f"{v:,.0f} N".replace(",", " ")

# fmt_lon/fmt_lat y generar_ticks (en grados) se mantienen para el Mapa 2
# (grilla PISCO), que sigue en lat/lon porque sus pixeles son nativos en
# grados; solo el Mapa 1 (ubicacion de estaciones) pasa a UTM zona 17S.
def fmt_lon(v):
    return f"{abs(v):.1f}°O"

def fmt_lat(v):
    return f"{abs(v):.1f}°S"

def generar_ticks_utm(vmin_m, vmax_m, paso_m):
    """Igual que antes pero en metros UTM: valores de grilla cada `paso_m`
    metros, contando en pasos enteros para incluir ambos extremos sin el
    arrastre de error de punto flotante de np.arange."""
    n0 = round(vmin_m / paso_m)
    n1 = round(vmax_m / paso_m)
    return np.arange(n0, n1 + 1) * paso_m

def generar_ticks(vmin, vmax, paso):
    """Version en grados (lat/lon) de generar_ticks_utm, usada por el Mapa 2
    (grilla PISCO). Misma logica de conteo en pasos enteros para evitar el
    arrastre de error de punto flotante de np.arange con pasos como 0.1."""
    n0 = round(vmin / paso)
    n1 = round(vmax / paso)
    return np.round(np.arange(n0, n1 + 1) * paso, 10)

paso = 0.1  # grados, usado por el Mapa 2

# Esquinas del recorte geografico en UTM, para saber el rango Este/Norte.
e00, n00 = to_utm.transform(xmin1, ymin1)
e01, n01 = to_utm.transform(xmin1, ymax1)
e10, n10 = to_utm.transform(xmax1, ymin1)
e11, n11 = to_utm.transform(xmax1, ymax1)
este_min, este_max = min(e00, e01), max(e10, e11)
norte_min, norte_max = min(n00, n10), max(n01, n11)

paso_m = 10000
lon_mid, lat_mid = (xmin1 + xmax1) / 2, (ymin1 + ymax1) / 2
este_mid, norte_mid = to_utm.transform(lon_mid, lat_mid)
este_ticks = generar_ticks_utm(este_min, este_max, paso_m)
norte_ticks = generar_ticks_utm(norte_min, norte_max, paso_m)

x_ticks_m = [transformer.transform(*from_utm.transform(e, norte_mid))[0] for e in este_ticks]
y_ticks_m = [transformer.transform(*from_utm.transform(este_mid, n))[1] for n in norte_ticks]

ax.set_xticks(x_ticks_m)
ax.set_xticklabels([fmt_este(v) for v in este_ticks], fontsize=7.3, fontweight="bold")
ax.set_yticks(y_ticks_m)
# Etiquetas Norte (verticales) giradas 90°, estilo cartografico profesional:
# en metros completos ("9 220 000 m N") no caben horizontales sin solaparse.
ax.set_yticklabels([fmt_norte(v) for v in norte_ticks], fontsize=7.3,
                    fontweight="bold", rotation=90, va="center")
# Etiquetas de coordenadas POR FUERA del mapa, en los cuatro lados, dentro de una
# banda blanca (estilo cartografico clasico): mapa -> numeros -> linea negra.
ax.tick_params(axis="both", direction="out", length=3, pad=4, color="black",
                labelcolor="#1a1a1a", top=True, right=True,
                labeltop=True, labelright=True)
ax.tick_params(axis="y", pad=6)
ax.grid(True, color="white", linewidth=0.7, alpha=0.6, linestyle="-", zorder=2)
ax.set_xlim(lim_x0, lim_x1)
ax.set_ylim(lim_y0, lim_y1)

for spine in ax.spines.values():
    spine.set_visible(True); spine.set_edgecolor("black"); spine.set_linewidth(0.8)

# Barra de escala aproximada (5 km) usando la proyeccion Web Mercator local.
# Anclada en la esquina inferior DERECHA (sin estaciones cerca) en vez de la
# izquierda, porque con el recorte mas ajustado del mapa la esquina inferior
# izquierda queda demasiado cerca de la etiqueta de Asuncion.
_, y0 = transformer.transform(xmin1 + 0.02, ymin1 + 0.03)
x1, _ = transformer.transform(xmax1 - 0.03, ymin1 + 0.03)
metros_por_grado = x1 - transformer.transform(xmax1 - 1.03, ymin1 + 0.03)[0]
x0 = x1 - 5000
ax.plot([x0, x1], [y0, y0], color="black", linewidth=2.5, solid_capstyle="butt")
ax.annotate("5 km", xy=((x0 + x1) / 2, y0), xycoords="data", xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

agregar_flecha_norte(ax, fig)

# Leyenda del marcador, con el titulo del sistema de referencia como primera
# linea del mismo recuadro (fondo blanco, borde negro) para que ambos textos
# se distingan con claridad del mapa de fondo, en la esquina inferior
# izquierda (sin estaciones ni barra de escala cerca).
leyenda_ancho_in, leyenda_alto_in = 2.55, 0.48
leyenda_margen_in = 0.09
lx0 = leyenda_margen_in / ancho_mapa_in
ly0 = leyenda_margen_in / alto_mapa_in
lw = leyenda_ancho_in / ancho_mapa_in
lh = leyenda_alto_in / alto_mapa_in
caja_leyenda = FancyBboxPatch(
    (lx0, ly0), lw, lh, transform=ax.transAxes,
    boxstyle="round,pad=0.01,rounding_size=0.012",
    facecolor="white", edgecolor=COLOR_MUTED, linewidth=0.7,
    alpha=0.96, zorder=10, clip_on=False,
)
ax.add_patch(caja_leyenda)
leyenda_x = lx0 + (0.22 / ancho_mapa_in)
fila_titulo_y = ly0 + lh - (0.15 / alto_mapa_in)
fila_marcador_y = ly0 + (0.15 / alto_mapa_in)
ax.annotate(
    "Coordenadas UTM Zona 17S, WGS84 (m)", xy=(lx0 + (0.10 / ancho_mapa_in), fila_titulo_y),
    xycoords="axes fraction", xytext=(0, 0), textcoords="offset points",
    fontsize=7, style="italic", color="#1a1a1a", ha="left", va="center",
    zorder=11,
)
ax.scatter([leyenda_x], [fila_marcador_y], s=45, color="black", edgecolor="white",
           linewidth=1.3, transform=ax.transAxes, zorder=11, clip_on=False)
ax.annotate(
    "Estación pluviométrica SENAMHI", xy=(leyenda_x, fila_marcador_y),
    xycoords="axes fraction", xytext=(10, 0), textcoords="offset points",
    fontsize=7.3, fontweight="bold", color="#1a1a1a", ha="left", va="center",
    zorder=11,
)

# Marco exterior negro que encierra el mapa y la banda blanca con los numeros.
from matplotlib.patches import Rectangle
e = 0.04  # separacion al borde de la figura, en pulgadas
fig.add_artist(Rectangle((e / ancho_total_in, e / alto_total_in),
                          1 - 2 * e / ancho_total_in, 1 - 2 * e / alto_total_in,
                          transform=fig.transFigure, fill=False, edgecolor="black",
                          linewidth=1.4, zorder=20))

guardar(fig, "mapa_ubicacion_estaciones")

# ---------------------------------------------------------------------------
# MAPA 2: Grilla PISCO real con pixeles de vecino mas cercano
# ---------------------------------------------------------------------------
print("Abriendo NetCDF PISCO (lazy)...")
ds = xr.open_dataset(f"{BASE}/datos/PISCOp_dv03.nc")
lon_name = "longitude" if "longitude" in ds.coords else "lon"
lat_name = "latitude" if "latitude" in ds.coords else "lat"

sub = ds.sel({lon_name: slice(xmin - 0.15, xmax + 0.15), lat_name: slice(ymax + 0.15, ymin - 0.15)})
if sub[lat_name].values[0] < sub[lat_name].values[-1]:
    sub = ds.sel({lon_name: slice(xmin - 0.15, xmax + 0.15), lat_name: slice(ymin - 0.15, ymax + 0.15)})

precip_media = sub["precipitation"].isel(Z1=slice(0, 365)).mean(dim="Z1").values
lons_grid = sub[lon_name].values
lats_grid = sub[lat_name].values

fig, ax = plt.subplots(figsize=(6.6, 6.0))
malla = ax.pcolormesh(lons_grid, lats_grid, precip_media, cmap="YlGnBu", shading="auto",
                       vmin=0, edgecolors=COLOR_MUTED, linewidth=0.15)
cbar = fig.colorbar(malla, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Precipitación media diaria PISCO, año 1 (mm/día)", fontsize=8)

res_lon = abs(lons_grid[1] - lons_grid[0]) if len(lons_grid) > 1 else 0.1
res_lat = abs(lats_grid[1] - lats_grid[0]) if len(lats_grid) > 1 else 0.1

for est, (lon, lat, alt) in ESTACIONES_COORDS.items():
    idx_lon = np.argmin(np.abs(lons_grid - lon))
    idx_lat = np.argmin(np.abs(lats_grid - lat))
    px_lon, px_lat = lons_grid[idx_lon], lats_grid[idx_lat]
    ax.add_patch(plt.Rectangle((px_lon - res_lon/2, px_lat - res_lat/2), res_lon, res_lat,
                                fill=False, edgecolor=COLOR_XGB, linewidth=1.8, zorder=5))
    ax.scatter(lon, lat, s=45, color="black", marker="o", zorder=6, edgecolor="white", linewidth=0.8)
    ax.annotate(ESTACIONES_LABEL[est], (lon, lat), fontsize=7.3, xytext=(5, 4),
                textcoords="offset points", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Grilla de coordenadas (graticula), en el mismo formato °O/°S del mapa 1,
# para presentacion consistente entre ambas figuras.
xlim, ylim = ax.get_xlim(), ax.get_ylim()
lon_ticks2 = generar_ticks(xlim[0], xlim[1], paso)
lat_ticks2 = generar_ticks(ylim[0], ylim[1], paso)
ax.set_xticks(lon_ticks2)
ax.set_xticklabels([fmt_lon(v) for v in lon_ticks2], fontsize=7.3)
ax.set_yticks(lat_ticks2)
ax.set_yticklabels([fmt_lat(v) for v in lat_ticks2], fontsize=7.3)
ax.grid(True, color="white", linewidth=0.5, alpha=0.5, linestyle="-", zorder=2)

# Barra de escala aproximada (5 km), calculada en grados a partir de la
# latitud media del recuadro (aqui los ejes estan en lon/lat, no en Web
# Mercator como el mapa 1).
lat_media = (ylim[0] + ylim[1]) / 2
km_por_grado_lon = 111.32 * np.cos(np.radians(lat_media))
deg_5km = 5 / km_por_grado_lon
x0b, y0b = xlim[0] + 0.03, ylim[0] + 0.03
ax.plot([x0b, x0b + deg_5km], [y0b, y0b], color="black", linewidth=2.5,
        solid_capstyle="butt", zorder=7)
ax.annotate("5 km", xy=(x0b + deg_5km / 2, y0b), xycoords="data", xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=7)

for spine in ax.spines.values():
    spine.set_visible(True); spine.set_edgecolor(COLOR_MUTED); spine.set_linewidth(0.8)

ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_title("Grilla del producto PISCOp v03 y píxel de vecino más cercano por estación",
             loc="left", fontweight="bold", fontsize=10)
from matplotlib.lines import Line2D
leyenda = [Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markeredgecolor="white", markersize=7, label="Estación SENAMHI"),
           Line2D([0], [0], color=COLOR_XGB, linewidth=1.8, label="Píxel PISCO seleccionado")]
ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
          frameon=False, fontsize=7.5)
fig.tight_layout()
agregar_flecha_norte(ax, fig)
guardar(fig, "mapa_grilla_pisco")

print(f"Resolucion aproximada de la grilla: {res_lon:.3f}° lon x {res_lat:.3f}° lat")
print("Listo.")
