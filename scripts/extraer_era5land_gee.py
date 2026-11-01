#!/usr/bin/env python3
"""Extrae series diarias de ERA5-Land (temperatura, viento, presión, humedad de
suelo, radiación) y deriva humedad relativa (Magnus/Tetens), vía Google Earth
Engine, para las 6 estaciones de la tesis.

No extrae variables topográficas (elevación/pendiente/orientación, SRTM):
descartadas por indicación del asesor, por no considerarlas influyentes
para un dato puntual de estación.

Requiere autenticación previa (una sola vez por máquina):
    earthengine authenticate
o, dentro de Python:
    import ee; ee.Authenticate()

Uso:
    python3 scripts/extraer_era5land_gee.py

Salida:
    datos/era5land_estaciones.csv   (una fila por estación y día)
"""
from pathlib import Path
import time

import ee
import numpy as np
import pandas as pd

# --- Inicialización ---
# reemplaza 'tu-project-id' por el Project ID que creaste en Google Cloud
# (ver guía: console.cloud.google.com -> nombre del proyecto en la barra superior)
PROJECT_ID = "tesis-lstm-xgboost"
ee.Initialize(project=PROJECT_ID)

BASE = str(Path(__file__).resolve().parent.parent)
OUT = f"{BASE}/datos"

# Mismas coordenadas que scripts/generar_mapas.py (lon, lat, altitud SENAMHI)
ESTACIONES_COORDS = {
    "AUGUSTO_WEBERBAUER": (-78.49, -7.17, 2673),
    "ASUNCION": (-78.52, -7.33, 2270),
    "JESUS": (-78.39, -7.25, 2564),
    "LA_ENCANADA": (-78.33, -7.12, 2980),
    "NAMORA": (-78.33, -7.20, 2760),
    "SAN_JUAN": (-78.49, -7.30, 2253),
}

FECHA_INICIO = "2000-01-01"
FECHA_FIN = "2026-01-01"  # exclusivo: cubre hasta 2025-12-31

# Bandas de ERA5-Land Daily Aggregated relevantes para este caso. Lista completa
# en https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR
# Nota: ERA5-Land entrega las 4 bandas de temperatura en Kelvin; el CSV final
# las guarda ya convertidas a Celsius (ver convertir_temperaturas_a_celsius).
BANDAS_ERA5 = [
    "temperature_2m",                      # K en origen -> °C en el CSV final, media diaria
    "temperature_2m_min",                  # K en origen -> °C en el CSV final
    "temperature_2m_max",                  # K en origen -> °C en el CSV final
    "dewpoint_temperature_2m",             # K en origen -> °C en el CSV final
    "u_component_of_wind_10m",             # m/s
    "v_component_of_wind_10m",             # m/s
    "surface_pressure",                    # Pa
    "surface_net_solar_radiation_sum",     # J/m^2 acumulado del día
    "volumetric_soil_water_layer_1",       # m^3/m^3, capa 0-7cm
]


def extraer_serie_estacion(nombre, lon, lat):
    """Extrae la serie diaria completa de ERA5-Land para un punto, un año a la
    vez, usando reduceRegion imagen por imagen (NO getRegion).

    getRegion(punto, scale) se probó primero y resultó en un bug real: para
    puntos cercanos entre si (ej. La Encañada y Namora, separados 8.9 km)
    devolvia las MISMAS coordenadas de pixel y, por tanto, la MISMA serie
    completa para ambas estaciones -- un reescalado/reproyeccion interno de
    getRegion que no respeta el pixel nativo cuando el `scale` pedido no
    coincide exactamente con como esa funcion arma la grilla de muestreo.
    reduceRegion(ee.Reducer.first(), punto, scale) sobre cada imagen si da
    el valor del pixel real que contiene el punto (confirmado contra
    ee.Image.sample(), que usa el pixel nativo exacto)."""
    punto = ee.Geometry.Point([lon, lat])
    coleccion = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(BANDAS_ERA5)

    filas = []
    anio_inicio = int(FECHA_INICIO[:4])
    anio_fin = int(FECHA_FIN[:4])
    for anio in range(anio_inicio, anio_fin):
        t0 = f"{anio}-01-01"
        t1 = f"{anio + 1}-01-01"
        sub = coleccion.filterDate(t0, t1).filterBounds(punto)

        def extraer_un_dia(img, punto=punto):
            valores = img.reduceRegion(ee.Reducer.first(), punto, scale=11132)
            return ee.Feature(None, valores.set("fecha", img.date().format("YYYY-MM-dd")))

        fc = ee.FeatureCollection(sub.map(extraer_un_dia))
        resultado = fc.getInfo()
        registros = [f["properties"] for f in resultado["features"]]
        for r in registros:
            filas.append({"ESTACION": nombre, "FECHA": r["fecha"], **{b: r[b] for b in BANDAS_ERA5}})
        print(f"  {nombre} {anio}: {len(registros)} días extraídos")
        time.sleep(0.5)  # margen de cortesía frente a cuota de requests/segundo

    return pd.DataFrame(filas)


def agregar_humedad_relativa(df):
    """Humedad relativa (%) vía fórmula de Magnus/Tetens, a partir de la
    temperatura y el punto de rocío a 2m que ya trae ERA5-Land (ambas en
    Kelvin, tal como las entrega la colección). No requiere ninguna descarga
    adicional.

    RH = 100 * exp(a*Td/(b+Td)) / exp(a*T/(b+T)), con a=17.625, b=243.04 °C
    (constantes de la aproximación de Magnus; Td y T en Celsius).

    Advertencia a documentar en la tesis: Bonshoms et al. (2022, Journal of
    Mountain Science, DOI 10.1007/s11629-022-7388-4) validaron ERA5-Land
    contra observaciones en glaciares peruanos y encontraron que su fidelidad
    para humedad relativa es notablemente menor que para temperatura en los
    Andes (r~0.4-0.6 para HR, frente a r>0.80 para temperatura) -- el ruido
    de esa variable base se propaga a esta derivada."""
    t_c = df["temperature_2m"] - 273.15
    td_c = df["dewpoint_temperature_2m"] - 273.15
    a, b = 17.625, 243.04
    es_t = np.exp((a * t_c) / (b + t_c))
    es_td = np.exp((a * td_c) / (b + td_c))
    df["humedad_relativa_pct"] = 100 * (es_td / es_t)
    return df


def convertir_temperaturas_a_celsius(df):
    """Convierte a Celsius las 4 columnas de temperatura que ERA5-Land entrega
    en Kelvin, para que coincidan con la unidad de uso local (Cajamarca) en
    cualquier tabla, figura o reporte que lea directamente de este CSV.

    No afecta el entrenamiento: RobustScaler/MinMaxScaler son transformaciones
    afines, así que escalar en Kelvin o en Celsius produce el mismo tensor
    final (restar 273.15 a toda la serie no cambia su dispersión relativa)."""
    for col in ["temperature_2m", "temperature_2m_min", "temperature_2m_max", "dewpoint_temperature_2m"]:
        df[col] = df[col] - 273.15
    return df


def main():
    print("--- Extrayendo ERA5-Land (diario, 2000-2025) ---")
    dfs = []
    for nombre, (lon, lat, _alt) in ESTACIONES_COORDS.items():
        print(f"Estación: {nombre} ({lon}, {lat})")
        dfs.append(extraer_serie_estacion(nombre, lon, lat))
    df_era5 = pd.concat(dfs, ignore_index=True)
    df_era5 = agregar_humedad_relativa(df_era5)  # calculada antes de convertir, usa Kelvin internamente
    df_era5 = convertir_temperaturas_a_celsius(df_era5)
    df_era5.to_csv(f"{OUT}/era5land_estaciones.csv", index=False)
    print(f"Guardado: {OUT}/era5land_estaciones.csv ({len(df_era5)} filas, temperaturas en °C)")


if __name__ == "__main__":
    main()
