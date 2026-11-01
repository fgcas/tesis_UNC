# Reconstrucción de precipitación diaria en Cajamarca — LSTM vs. XGBoost

Código y notebook de la tesis de pregrado *"Análisis comparativo de Redes Neuronales Recurrentes (LSTM) y algoritmos de Gradient Boosting (XGBoost) integrando datos satelitales para la reconstrucción de precipitaciones diarias en la provincia de Cajamarca"* (Escuela Profesional de Ingeniería Hidráulica, Universidad Nacional de Cajamarca).

**Autor:** Frank Gianmarco Casanova Quiroz
**Asesor:** Dr. Ing. José Hilario Longa Álvarez

Este repositorio contiene únicamente lo necesario para **reproducir los resultados** de la tesis: no incluye el documento de la tesis en sí, ni materiales de sustentación.

## Qué hay en este repositorio

- `notebook/001_Tesis_FGCQ.ipynb` — notebook principal, pensado para ejecutarse de corrido en Google Colab. Limpieza y análisis exploratorio del registro de SENAMHI, imputación multivariada (MICE), extracción de PISCO, construcción del tensor de entrada (ventanas de 15 días), entrenamiento y evaluación de la LSTM y de XGBoost.
- `scripts/` — scripts de generación de las figuras de resultados, y la extracción de variables de ERA5-Land vía Google Earth Engine.
- `tesis/figuras_data/` — CSV/NPZ/JSON intermedios que el notebook exporta (métricas, predicciones, importancia de variables) y que `scripts/` usa para generar las figuras finales, sin necesidad de re-ejecutar el notebook completo.
- `requirements.txt` — dependencias para ejecución local (fuera de Colab).

## Qué NO hay en este repositorio (y por qué)

| Archivo | Motivo | Cómo obtenerlo |
|---|---|---|
| `datos/DATA_PP.csv` | Registro diario de precipitación de SENAMHI, entregado mediante solicitud institucional directa (ver Anexo E de la tesis). No es un dato de redistribución libre. | Solicitar a SENAMHI (https://www.senamhi.gob.pe/) el registro diario de las estaciones de interés. |
| `datos/PISCOp_dv03.nc` | Público y gratuito, pero pesa ~1.5 GB — demasiado para un repositorio de git. | Descargar `PISCOp_d.nc` del dataset *PISCOp v3.0* (Gutiérrez & Lavado-Casimiro, 2025), DOI [10.6084/m9.figshare.32411886](https://doi.org/10.6084/m9.figshare.32411886), y renombrarlo a `PISCOp_dv03.nc`. |

El notebook trae, en su segunda celda, las instrucciones completas de instalación y de dónde colocar estos dos archivos antes de ejecutar.

## Cómo ejecutar el notebook

1. Abrir `notebook/001_Tesis_FGCQ.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Subir (o montar desde Drive) `datos/DATA_PP.csv` (propio) y `datos/PISCOp_dv03.nc` (descargado, ver tabla arriba), manteniendo la estructura `datos/` junto al notebook.
3. Ejecutar todas las celdas en orden (Entorno de ejecución → Ejecutar todas). El entrenamiento completo (búsqueda de hiperparámetros + modelos finales de LSTM y XGBoost) toma entre 30 y 90 minutos según el hardware disponible en la sesión de Colab.

Las variables de ERA5-Land (humedad relativa, temperatura mínima, viento) no se extraen en este notebook: se calculan aparte, vía Google Earth Engine, en `scripts/extraer_era5land_gee.py`.

## Panel interactivo de resultados

Un recorrido visual del método y los resultados, sin necesidad de ejecutar nada: **[ver panel interactivo](https://claude.ai/artifact/2gbN4NWiGdBcXz4um4Bgdu)**.

## Fuentes de datos y licencias

- **SENAMHI** — registro pluviométrico observado, Dirección Zonal 3 (Cajamarca). Uso autorizado para esta investigación, no redistribuido.
- **PISCOp v3.0** — Gutiérrez, L. & Lavado-Casimiro, W. (2025). *PISCOp (v3.0): Actualización de datos grillados de precipitación.* SENAMHI. Dataset: [10.6084/m9.figshare.32411886](https://doi.org/10.6084/m9.figshare.32411886).
- **ERA5-Land** — Muñoz-Sabater, J. et al. (2021), Copernicus Climate Change Service (C3S) / ECMWF.
- **Mapas base** — © OpenStreetMap contributors, datos bajo licencia [ODbL](https://opendatacommons.org/licenses/odbl/).

## Cita

Si este trabajo te resulta útil, puedes citarlo como:

> Casanova Quiroz, F. G. (2026). *Análisis comparativo de Redes Neuronales Recurrentes (LSTM) y algoritmos de Gradient Boosting (XGBoost) integrando datos satelitales para la reconstrucción de precipitaciones diarias en la provincia de Cajamarca* [Tesis de pregrado, Universidad Nacional de Cajamarca].
