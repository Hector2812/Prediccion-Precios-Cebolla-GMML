# Pronóstico diario del precio mayorista de cebolla cabeza roja en el GMML

Proyecto de investigación del curso **Seminario de Investigación 2** de la **Universidad de Lima**. Compara modelos estadísticos, de aprendizaje automático, de aprendizaje profundo y un modelo híbrido para pronosticar el **precio promedio mayorista diario de la cebolla cabeza roja** comercializada en el **Gran Mercado Mayorista de Lima (GMML)**.

Los datos proceden del **Sistema de Información de Abastecimiento y Precios (SISAP)** del Ministerio de Desarrollo Agrario y Riego (MIDAGRI), para el periodo **2012-2026**. La variable objetivo se expresa en **soles por kilogramo (S/kg)** y el horizonte de pronóstico es de **un día (`t+1`)**.

> **Alcance:** evaluación retrospectiva. El procesamiento evita usar observaciones futuras para construir los predictores, pero no se ha verificado la hora histórica de publicación de cada reporte de SISAP. Por ello, los resultados no deben interpretarse como una validación operativa en tiempo real.

## 1. Archivos del proyecto

El código se organiza en **seis archivos Python**. La instalación utiliza **un único archivo de dependencias**, `requirements.txt`.

```text
Proyecto_Prediccion_Cebolla/
├── fase1_adquisicion_integracion.py
├── fase2_preprocesamiento_ingenieria.py
├── fase3_implementacion_modelos.py
├── fase3b_ablacion.py
├── fase4_evaluacion_comparacion.py
├── fase4b_ablacion.py
├── requirements.txt
└── README.md
```

| Archivo | Función |
|---|---|
| `fase1_adquisicion_integracion.py` | Descarga reportes de precios y abastecimiento desde SISAP, unifica las fechas y genera el conjunto diario consolidado. |
| `fase2_preprocesamiento_ingenieria.py` | Audita y trata los faltantes, prepara los predictores temporales y construye el objetivo de pronóstico a un día. |
| `fase3_implementacion_modelos.py` | Entrena y valida los modelos principales con particiones temporales expansivas; selecciona configuraciones e integra la ablación de la Fase 3. |
| `fase3b_ablacion.py` | Entrena y valida los escenarios complementarios que excluyen los predictores derivados del precio histórico. |
| `fase4_evaluacion_comparacion.py` | Ejecuta la prueba final, calcula métricas, contrastes estadísticos y explicaciones de los modelos; integra la ablación de la Fase 4. |
| `fase4b_ablacion.py` | Evalúa los escenarios de ablación, sus contrastes estadísticos y los análisis de explicabilidad complementarios. |
| `requirements.txt` | Dependencias Python del proyecto completo. |

Las carpetas de **datos, modelos entrenados, resultados, figuras y demás evidencias** se conservan en el repositorio. La estructura anterior destaca únicamente los archivos de código y la instalación; no enumera cada archivo generado.

Los módulos `fase3b_ablacion.py` y `fase4b_ablacion.py` se ejecutan desde las fases principales. **Para repetir el flujo completo, basta con ejecutar las fases 1, 2, 3 y 4 en ese orden.**

## 2. Modelos y conjuntos de información

**Modelos evaluados:** SARIMAX, XGBoost, LSTM, TCN y LSTM-XGBoost (LSTM con corrector residual XGBoost). Las referencias son **Naive** y **Seasonal Naive** de siete días.

Se comparan dos conjuntos principales:

- **Histórico:** precio histórico y variables calendario.
- **Histórico + abastecimiento:** histórico del precio, calendario y volumen de ingreso.

Como **análisis exploratorio complementario de ablación**, XGBoost, LSTM, TCN y LSTM-XGBoost también se reentrenan con:

- **Calendario:** sin predictores derivados del precio histórico.
- **Calendario + abastecimiento:** calendario y volumen, sin predictores derivados del precio histórico.

La ablación se incorporó como análisis **post hoc** y no constituye una hipótesis confirmatoria establecida antes de observar los resultados principales.

## 3. Requisitos e instalación

La ejecución incluida en el proyecto registra **Windows**, **Python 3.14.4** y **PyTorch 2.11.0+cu130 (CUDA 13.0)**. Las principales versiones de bibliotecas están indicadas en `requirements.txt`. Algunas dependencias no tienen una versión fijada porque no se registró ese dato en los metadatos de ejecución.

**Instalación en PowerShell**, desde la carpeta principal del proyecto:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para reproducir la aceleración por GPU se necesita una instalación de PyTorch y CUDA compatible con el equipo. Consulta las instrucciones oficiales de [PyTorch](https://pytorch.org/get-started/locally/). La disponibilidad de una GPU no es obligatoria para todos los pasos, pero puede influir considerablemente en el tiempo de entrenamiento. No se ha validado la instalación íntegra en un entorno limpio distinto al original.

La **Fase 1** requiere **Microsoft Edge**, acceso a Internet y Selenium para descargar los reportes de SISAP. Los `.xls` de este proyecto contienen tablas HTML que se procesan mediante `pandas.read_html`.

## 4. Ejecución

Abre una terminal en la raíz del proyecto y ejecuta:

```powershell
python fase1_adquisicion_integracion.py
python fase2_preprocesamiento_ingenieria.py
python fase3_implementacion_modelos.py
python fase4_evaluacion_comparacion.py
```

- **Fase 1:** descarga e integración de datos.
- **Fase 2:** preprocesamiento y construcción del conjunto de modelado.
- **Fase 3:** entrenamiento, validación temporal, selección de hiperparámetros y ablación.
- **Fase 4:** evaluación final, comparación estadística, explicabilidad y resultados de ablación.

Si ya existe el conjunto diario consolidado, puede comenzarse desde la **Fase 2**. Si ya existe el conjunto procesado y su protocolo correspondiente, puede comenzarse desde la **Fase 3**. Para ejecutar únicamente la evaluación final es necesario conservar también los artefactos de los modelos entrenados.

**Precaución:** una nueva ejecución puede sobrescribir los resultados guardados. Conviene conservar una copia de la corrida original antes de repetir los experimentos.

## 5. Datos y evaluación

- **Periodo de origen:** 1 de enero de 2012 al 1 de septiembre de 2026.
- **Observaciones consolidadas:** 5 358 fechas.
- **Pares de origen y objetivo `t+1`:** 5 357.
- **Objetivos sin observación real:** 42; no se imputan como etiquetas reales para entrenamiento o evaluación.
- **Tratamiento causal de predictores faltantes:** Forward Fill para precio de entrada y EWMA para abastecimiento de entrada; selección de **α = 0,10** en desarrollo.
- **Partición temporal:** 80 % de implementación y 20 % de prueba final.
- **Validación interna:** `TimeSeriesSplit` con cinco folds de entrenamiento expansivo.
- **Prueba final:** 1 072 días, del 26 de septiembre de 2023 al 1 de septiembre de 2026.
- **Métrica principal:** sMAPE. Métricas adicionales: MAE, RMSE, MASE y R².
- **Comparación estadística:** prueba de Diebold-Mariano con pérdida cuadrática y ajuste de Holm (`α = 0,05`).
- **Explicabilidad:** coeficientes de SARIMAX; TreeSHAP para XGBoost y el corrector híbrido; TimeSHAP para LSTM; Integrated Gradients para TCN.

Los pronósticos se construyen con la información asociada al día de origen `t` para estimar el precio de `t+1`. La disponibilidad histórica efectiva de los datos de SISAP es una limitación pendiente de comprobar.

## 6. Resultados de la ejecución incluida

Valores de sMAPE de la prueba final, extraídos de los archivos de resultados presentes en el proyecto. **Un valor menor significa menor error.**

| Modelo | Histórico (%) | Histórico + abastecimiento (%) |
|---|---:|---:|
| SARIMAX | 3,069 | 3,071 |
| XGBoost | 3,432 | 3,426 |
| LSTM | 3,440 | 3,379 |
| TCN | 3,854 | 6,852 |
| LSTM-XGBoost | 3,798 | 3,815 |

**Referencias:** Naive = **3,069 %** y Seasonal Naive = **11,562 %**. SARIMAX-Histórico reprodujo los pronósticos de Naive en esta prueba. Ningún modelo demostró una mejora estadísticamente significativa frente a Naive bajo los contrastes establecidos.

**Ablación exploratoria:** al retirar los predictores derivados del historial del precio, los errores sMAPE aumentaron considerablemente en los cuatro modelos reentrenados. El análisis señala la relevancia predictiva del precio histórico bajo las configuraciones evaluadas, no una relación causal sobre la formación de precios.

## 7. Datos, modelos y evidencias guardadas

El proyecto conserva las carpetas originales y los productos de cada fase:

```text
data descargada/        Reportes originales de SISAP
data consolidada/       Datos diarios integrados y metadatos
data procesada/         Datos preparados para los modelos y protocolo
outputs/validacion/    Entrenamientos, hiperparámetros y modelos ajustados
outputs/evaluacion_final/  Pronósticos, métricas, pruebas estadísticas y explicabilidad
outputs/tablas_experimentacion/  Tablas de los procedimientos experimentales
outputs/tablas_resultados/       Tablas de evaluación
Figuras/experimentacion/        Figuras del tratamiento de datos y entrenamiento
Figuras/resultados_discusion/   Figuras de desempeño e interpretación
```

Los directorios anteriores son los utilizados por los scripts. Se recomienda conservar los nombres exactos, incluidas mayúsculas y espacios. Los archivos de salida permiten revisar resultados sin volver a entrenar todos los modelos.

## 8. Fuente de datos y uso académico

**Fuente:** [Sistema de Información de Abastecimiento y Precios (SISAP), MIDAGRI](https://sistemas.midagri.gob.pe/sisap/portal2/mayorista/).

Antes de publicar los datos descargados, verifica las condiciones de reutilización y redistribución del portal. Revisa también que ningún archivo o metadato revele datos privados o credenciales. El repositorio acompaña una investigación académica y los resultados se limitan al mercado, horizonte y condiciones experimentales indicados.
