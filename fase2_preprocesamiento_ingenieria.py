from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys
import traceback
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "Fase_2_log.txt"
INPUT_FILE = ROOT / "data consolidada" / "dataset_diario_consolidado.csv"
OUTPUT_DIR = ROOT / "data procesada"
OUTPUT_FILE = OUTPUT_DIR / "dataset_modelado_base.csv"
FIGURE_DIR = ROOT / "Figuras" / "experimentacion"
TABLAS_DIR = ROOT / "outputs" / "tablas_experimentacion"

DATE = "fecha"
TARGET_DATE = "fecha_target"
PRICE = "precio_promedio_soles_kg"
VOLUME = "volumen_toneladas"
TARGET = "precio_objetivo_t1"


EWMA_ALPHA: float | None = None
ALFAS_SENSIBILIDAD = (0.10, 0.20, 0.30, 0.40, 0.50)
PROPORCION_DESARROLLO = 0.80
PREPROCESSING_VERSION = "Fase2_preprocesamiento_causal_v4_seleccion_alpha"
SEED_SENSIBILIDAD = 42
BLOQUES_SENSIBILIDAD = ((1, 120), (5, 40), (21, 12))
CRITERIO_SELECCION_ALPHA = "minimo_MAE_promedio_equilibrado_por_duracion"
PRICE_OBS = "precio_observado_original"
VOLUME_OBS = "abastecimiento_observado_original"
PRICE_FLAG = "precio_imputado"
VOLUME_FLAG = "abastecimiento_imputado"
TARGET_FLAG = "objetivo_observado"

PRICE_LAGS = (1, 2, 3, 4, 5, 7, 14, 21, 28, 30)
VOLUME_LAGS = (1, 7, 14, 30)
MOVING_AVERAGE_WINDOWS = (7, 30)
MOVING_RANGE_WINDOWS = (30,)

CALENDAR_COLUMNS = (
    "dia_semana_sin",
    "dia_semana_cos",
    "mes_sin",
    "mes_cos",
    "dia_anio_sin",
    "dia_anio_cos",
)


def validar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = [DATE, PRICE, VOLUME]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.loc[:, required].copy()
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")

    if df[DATE].isna().any():
        raise ValueError("Se detectaron fechas no válidas.")

    df = df.sort_values(DATE).reset_index(drop=True)

    if df[DATE].duplicated().any():
        raise ValueError("Se detectaron fechas duplicadas.")

    if len(df) < 2:
        raise ValueError("El dataset no contiene observaciones suficientes.")

    diferencias = df[DATE].diff().dropna()
    if not diferencias.eq(pd.Timedelta(days=1)).all():
        raise ValueError("El dataset no conserva un calendario diario continuo.")

    if (df[PRICE].dropna() <= 0).any():
        raise ValueError("Se detectaron precios observados no positivos.")

    if (df[VOLUME].dropna() < 0).any():
        raise ValueError(
            "Se detectaron valores observados de abastecimiento negativos."
        )

    return df


def imputar_series(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa SOLO predictores de forma causal, sin cambiar la etiqueta t+1.

    Precio: último valor observado (LOCF/ffill).
    Abastecimiento: EWMA de valores OBSERVADOS previos, con alpha seleccionado.
    En un día ausente, el estado EWMA no se actualiza con el dato imputado.
    """
    if EWMA_ALPHA is None or not (0.0 < EWMA_ALPHA <= 1.0):
        raise RuntimeError("Debe seleccionar un alpha EWMA válido antes de imputar.")
    df = df.copy()
    if TARGET not in df.columns:
        raise ValueError("Primero construir_objetivo sobre datos originales.")

    df[PRICE_OBS] = df[PRICE].copy()
    df[VOLUME_OBS] = df[VOLUME].copy()
    df[PRICE_FLAG] = df[PRICE_OBS].isna().astype('int8')
    df[VOLUME_FLAG] = df[VOLUME_OBS].isna().astype('int8')
    df[TARGET_FLAG] = df[TARGET].notna().astype('int8')

    df[PRICE] = df[PRICE_OBS].ffill()

    ewma_pasado = df[VOLUME_OBS].ewm(
        alpha=EWMA_ALPHA, adjust=False, ignore_na=True
    ).mean()
    df[VOLUME] = df[VOLUME_OBS].where(df[VOLUME_OBS].notna(), ewma_pasado)

    for original, output in (
        (PRICE_OBS, "dias_sin_precio_observado"),
        (VOLUME_OBS, "dias_sin_abastecimiento_observado"),
    ):
        fecha_ultima = df[DATE].where(df[original].notna()).ffill()
        df[output] = (df[DATE] - fecha_ultima).dt.days

    if df[[PRICE, VOLUME]].isna().any().any():
        raise ValueError(
            "Persisten entradas faltantes; puede faltar un valor real al inicio del histórico. "
            "Debe establecerse una política inicial sin utilizar bfill."
        )
    if (df[PRICE] <= 0).any():
        raise ValueError("El precio imputado contiene valores no positivos.")
    if (df[VOLUME] < 0).any():
        raise ValueError("El abastecimiento imputado contiene valores negativos.")
    if not df.loc[df[PRICE_OBS].notna(), PRICE].equals(
        df.loc[df[PRICE_OBS].notna(), PRICE_OBS]
    ):
        raise ValueError("La imputación modificó precios realmente observados.")
    if not df.loc[df[VOLUME_OBS].notna(), VOLUME].equals(
        df.loc[df[VOLUME_OBS].notna(), VOLUME_OBS]
    ):
        raise ValueError("La imputación modificó volúmenes realmente observados.")

    return df


def construir_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_DATE] = df[DATE] + pd.Timedelta(days=1)
    df[TARGET] = df[PRICE].shift(-1)

    return df.iloc[:-1].reset_index(drop=True)


def codificar_calendario_ciclico(fechas: pd.Series) -> np.ndarray:
    fechas = pd.Series(pd.to_datetime(fechas, errors="coerce"))
    if fechas.isna().any():
        raise ValueError("No se pueden codificar fechas no válidas.")

    dia_semana = fechas.dt.dayofweek.to_numpy(dtype=float)
    mes = (fechas.dt.month.to_numpy(dtype=float) - 1.0)
    dia_anio = (fechas.dt.dayofyear.to_numpy(dtype=float) - 1.0)
    dias_anio = np.where(fechas.dt.is_leap_year.to_numpy(), 366.0, 365.0)

    return np.column_stack(
        [
            np.sin(2.0 * np.pi * dia_semana / 7.0),
            np.cos(2.0 * np.pi * dia_semana / 7.0),
            np.sin(2.0 * np.pi * mes / 12.0),
            np.cos(2.0 * np.pi * mes / 12.0),
            np.sin(2.0 * np.pi * dia_anio / dias_anio),
            np.cos(2.0 * np.pi * dia_anio / dias_anio),
        ]
    ).astype(float)


def construir_caracteristicas_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye seis variables calendario cíclicas a partir de fecha_target (t+1).
    Estas columnas se usan como covariables tabulares conocidas de antemano.
    Las redes secuenciales reconstruyen la misma codificación sobre cada fecha
    histórica de su ventana en las Fases 3 y 4.
    """
    df = df.copy()
    valores = codificar_calendario_ciclico(df[TARGET_DATE])
    for position, column in enumerate(CALENDAR_COLUMNS):
        df[column] = valores[:, position]
    return df


def construir_rezagos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in PRICE_LAGS:
        df[f"precio_lag_{lag}"] = df[PRICE].shift(lag - 1)

    for lag in VOLUME_LAGS:
        df[f"abastecimiento_lag_{lag}"] = df[VOLUME].shift(lag - 1)

    return df


def construir_caracteristicas_moviles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    for window in MOVING_AVERAGE_WINDOWS:
        df[f"precio_media_movil_{window}"] = (
            df[PRICE]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

    for window in MOVING_RANGE_WINDOWS:
        rolling = df[PRICE].rolling(
            window=window,
            min_periods=window,
        )
        df[f"precio_rango_movil_{window}"] = (
            rolling.max() - rolling.min()
        )

    return df


def construir_ventanas(
    df: pd.DataFrame,
    indices: np.ndarray,
    window: int,
    configuracion: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError(
            "La longitud de ventana debe ser mayor o igual que 1."
        )

    if configuracion not in {"H", "HA"}:
        raise ValueError(
            "La configuración debe ser 'H' o 'HA'."
        )

    price = df[PRICE].to_numpy(dtype=float)
    volume = df[VOLUME].to_numpy(dtype=float)
    target = df[TARGET].to_numpy(dtype=float)
    calendario_historico = codificar_calendario_ciclico(df[DATE])

    X: list[np.ndarray] = []
    y: list[float] = []
    rows: list[int] = []

    for i in np.asarray(indices, dtype=int):
        if i < window - 1 or i >= len(df):
            continue

        start = i - window + 1
        calendario_ventana = calendario_historico[start : i + 1]

        if configuracion == "H":
            values = np.column_stack(
                [
                    price[start : i + 1],
                    calendario_ventana,
                ]
            )
        else:
            values = np.column_stack(
                [
                    price[start : i + 1],
                    volume[start : i + 1],
                    calendario_ventana,
                ]
            )

        if not np.isfinite(values).all():
            continue

        if not np.isfinite(target[i]):
            continue

        X.append(values)
        y.append(float(target[i]))
        rows.append(int(i))

    if not X:
        n_features = 7 if configuracion == "H" else 8

        return (
            np.empty(
                (0, window, n_features),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=int,
            ),
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.float32,
        ),
        np.asarray(
            rows,
            dtype=int,
        ),
    )



def generar_evidencia_preprocesamiento(df: pd.DataFrame) -> None:
    """Tablas de auditoría y visuales descriptivos sin mirar predicciones finales."""
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    auditoria = []
    for variable, original, imputado, antiguedad in (
        ("Precio (S/kg)", PRICE_OBS, PRICE_FLAG, "dias_sin_precio_observado"),
        ("Abastecimiento (t)", VOLUME_OBS, VOLUME_FLAG, "dias_sin_abastecimiento_observado"),
    ):
        auditoria.append({
            "variable": variable, "dias_calendario_origen": len(df),
            "dias_observados": int(df[original].notna().sum()),
            "dias_predictor_imputado": int(df[imputado].sum()),
            "max_dias_desde_observacion": int(df[antiguedad].max()),
            "metodo": "LOCF" if original == PRICE_OBS else f"EWMA causal alfa={EWMA_ALPHA}",
            "objetivos_sin_precio_real": int((df[TARGET_FLAG] == 0).sum()),
            "objetivos_observados": int(df[TARGET_FLAG].sum()),
        })
    pd.DataFrame(auditoria).to_csv(TABLAS_DIR / "Tabla_2_auditoria_imputacion.csv", index=False, encoding="utf-8-sig")
    detalle = df.loc[(df[PRICE_FLAG] == 1) | (df[VOLUME_FLAG] == 1), [
        DATE, TARGET_DATE, PRICE_OBS, PRICE, VOLUME_OBS, VOLUME,
        PRICE_FLAG, VOLUME_FLAG, TARGET, TARGET_FLAG,
        "dias_sin_precio_observado", "dias_sin_abastecimiento_observado",
    ]].copy()
    detalle.to_csv(TABLAS_DIR / "Tabla_2b_detalle_imputaciones.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    catalogo = [{"caracteristica": c, "grupo": (
        "precio" if c.startswith("precio_") else
        "abastecimiento" if c.startswith("abastecimiento_") else
        "calendario" if c in CALENDAR_COLUMNS else "auditoria/objetivo"
    )} for c in df.columns]
    pd.DataFrame(catalogo).to_csv(TABLAS_DIR / "Tabla_2c_diccionario_variables.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6), sharex=True)
    for ax, observado, predictor, marca, titulo, unidad in [
        (axes[0], PRICE_OBS, PRICE, PRICE_FLAG, "Precio mayorista", "S/kg"),
        (axes[1], VOLUME_OBS, VOLUME, VOLUME_FLAG, "Abastecimiento", "t"),
    ]:
        ax.plot(df[DATE], df[predictor], lw=.75, alpha=.75, label="Entrada causal")
        ausentes = df[marca].eq(1)
        if ausentes.any():
            ax.scatter(df.loc[ausentes, DATE], df.loc[ausentes, predictor],
                       c="#B45309", s=18, zorder=3, label="Entrada imputada")
        ax.set_ylabel(f"{titulo} ({unidad})"); ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=.17)
    axes[-1].set_xlabel("Fecha de origen (t)")
    fig.suptitle("Observaciones e imputaciones temporales causales; objetivos no imputados")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figura_0b_series_y_puntos_imputados.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Tablas de imputación, diccionario de variables y figura de trazabilidad generadas.", flush=True)



def generar_figura_intervalo_faltante(df: pd.DataFrame) -> Path | None:
    """Evidencia del mayor intervalo ausente, sin imputar el objetivo real de t+1.

    El gráfico utiliza datos de la Fase 2 ya procesados. Las fechas reales
    permanecen como observaciones y los valores estimados se distinguen.
    """
    faltante = df[PRICE_FLAG].eq(1) & df[VOLUME_FLAG].eq(1)
    if not faltante.any():
        print("Sin ausencias conjuntas de precio y volumen para mostrar.", flush=True)
        return None
    grupo = (faltante.ne(faltante.shift(fill_value=False))).cumsum()
    tramos = (df.loc[faltante].groupby(grupo[faltante])
              .agg(inicio=(DATE, "min"), fin=(DATE, "max"), dias=(DATE, "size"))
              .sort_values(["dias", "inicio"], ascending=[False, True]))
    principal = tramos.iloc[0]
    inicio, fin = pd.Timestamp(principal.inicio), pd.Timestamp(principal.fin)
    margen = pd.Timedelta(days=12)
    ejemplo = df.loc[df[DATE].between(inicio-margen, fin+margen)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 6.8), sharex=True)
    for ax, original, predictor, marca, nombre, unidad in [
        (axes[0], PRICE_OBS, PRICE, PRICE_FLAG, "Precio mayorista", "S/kg"),
        (axes[1], VOLUME_OBS, VOLUME, VOLUME_FLAG, "Abastecimiento", "toneladas"),
    ]:
        ax.axvspan(inicio-pd.Timedelta(hours=12), fin+pd.Timedelta(hours=12),
                   color="#e5e7eb", alpha=0.75, label="Intervalo sin registro")
        ax.plot(ejemplo[DATE], ejemplo[original], "o-", markersize=3,
                color="#235789", lw=1.4, label="Dato observado")
        falta = ejemplo[marca].eq(1)
        # Mostrar sólo el tramo que se estimó y los dos puntos de anclaje.
        indices = np.flatnonzero(falta.to_numpy())
        if len(indices):
            viz = np.zeros(len(ejemplo), dtype=bool)
            for i in indices:
                viz[max(0, i-1):min(len(viz), i+2)] = True
            aproximacion = ejemplo[predictor].where(viz)
            ax.plot(ejemplo[DATE], aproximacion, "--", color="#bc6c25",
                    lw=1.7, label="Predictor completado (causal)")
            ax.scatter(ejemplo.loc[falta, DATE], ejemplo.loc[falta, predictor],
                       color="#bc6c25", s=14, zorder=4)
        ax.set_ylabel(f"{nombre} ({unidad})")
        ax.grid(alpha=0.16)
        ax.legend(loc="upper left", fontsize=8)
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    axes[-1].tick_params(axis="x", labelrotation=25)
    axes[-1].set_xlabel("Fecha de origen del pronóstico")
    fig.suptitle(
        f"Tratamiento causal del intervalo sin registros: {inicio:%d/%m/%Y} a {fin:%d/%m/%Y}"
        f" ({int(principal.dias)} días)", fontsize=12)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    destino = FIGURE_DIR / "Figura_articulo_tratamiento_faltantes_21_dias.png"
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)
    detalle = df.loc[df[DATE].between(inicio, fin), [
        DATE, TARGET_DATE, PRICE_OBS, PRICE, VOLUME_OBS, VOLUME,
        PRICE_FLAG, VOLUME_FLAG, TARGET, TARGET_FLAG,
    ]]
    detalle.to_csv(TABLAS_DIR / "Tabla_articulo_detalle_intervalo_faltante.csv",
                   index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(f"Figura de imputación para el artículo: {destino}", flush=True)
    return destino

def verificar_causalidad_y_rezagos(consolidado: pd.DataFrame, procesado: pd.DataFrame) -> None:
    """Pruebas de prefijos temporales, etiquetas reales y rezagos calendario."""
    assert len(procesado) == len(consolidado)-1
    esperado = consolidado[PRICE].shift(-1).iloc[:-1].to_numpy(dtype=float)
    actual = procesado[TARGET].to_numpy(dtype=float)
    if not np.array_equal(esperado,actual,equal_nan=True):
        raise AssertionError("Fuga: objetivo t+1 no coincide con precio REAL de t+1.")
    cortes=sorted(set([min(len(consolidado)-2,365),
                       min(len(consolidado)-2,1460),
                       min(len(consolidado)-2,3000),
                       min(len(consolidado)-2,int(len(consolidado)*.80))]))
    columnas=[PRICE,VOLUME,PRICE_FLAG,VOLUME_FLAG,
              "precio_lag_1","precio_lag_7", "abastecimiento_lag_1",
              "precio_media_movil_7", "precio_rango_movil_30"]
    for corte in cortes:
        base=consolidado.iloc[:corte+2].copy()
        breve=construir_objetivo(base)
        breve=imputar_series(breve)
        breve=construir_caracteristicas_calendario(breve)
        breve=construir_rezagos(breve)
        breve=construir_caracteristicas_moviles(breve)
        for col in columnas:
            a=procesado[col].iloc[:corte+1].to_numpy(dtype=float)
            b=breve[col].to_numpy(dtype=float)
            if not np.allclose(a,b,equal_nan=True,rtol=1e-10,atol=1e-10):
                raise AssertionError(f"Prueba de causalidad falló en {col} (corte={corte})")
    for lag in PRICE_LAGS:
        real=procesado[PRICE].shift(lag-1)
        if not np.allclose(procesado[f"precio_lag_{lag}"],real,equal_nan=True):
            raise AssertionError(f"Desfase incorrecto en precio_lag_{lag}")
    for lag in VOLUME_LAGS:
        real=procesado[VOLUME].shift(lag-1)
        if not np.allclose(procesado[f"abastecimiento_lag_{lag}"],real,equal_nan=True):
            raise AssertionError(f"Desfase incorrecto en abastecimiento_lag_{lag}")
    print(f"Pruebas de causalidad y rezagos: CORRECTAS ({len(cortes)} prefijos).")


def diagnostico_imputacion_con_mascaras(consolidado: pd.DataFrame) -> tuple[float, dict]:
    """Seleccionar EWMA alpha exclusivamente por reconstrucción del abastecimiento.

    El conjunto final de prueba se excluye antes de crear las máscaras. Todos
    los alfas utilizan los mismos bloques por duración. Selección por menor
    media NO ponderada de los tres MAE (igual peso a 1, 5 y 21 días).
    En empate numérico se selecciona el menor alpha. No se consulta sMAPE ni
    se ajusta un pronosticador; Fase 3 solo diagnostica sensibilidad predictiva.
    """
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    salida = TABLAS_DIR / "Tabla_2d_sensibilidad_imputacion_desarrollo.csv"
    seleccion_salida = TABLAS_DIR / "Tabla_2e_seleccion_alpha_EWMA.csv"
    json_salida = OUTPUT_DIR / "seleccion_alpha_ewma.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    n_origen = len(consolidado) - 1
    n_desarrollo = int(n_origen * PROPORCION_DESARROLLO)
    raw = consolidado.iloc[:n_desarrollo].copy().reset_index(drop=True)
    if len(raw) < 200:
        raise ValueError("No hay suficiente histórico para seleccionar alpha.")

    rng = np.random.default_rng(SEED_SENSIBILIDAD)
    registros = []
    detalle_bloques = {}
    for length, quantity in BLOQUES_SENSIBILIDAD:
        candidatos = [i for i in range(60, len(raw) - length - 1)
                      if raw[[PRICE, VOLUME]].iloc[i:i + length].notna().all().all()]
        rng.shuffle(candidatos)
        selected = []
        for pos in candidatos:
            if all(abs(pos - prev) > length + 30 for prev in selected):
                selected.append(pos)
            if len(selected) == quantity:
                break
        if not selected:
            raise RuntimeError(f"No existen bloques observados de {length} días para calibración EWMA.")
        selected.sort()
        detalle_bloques[str(length)] = [
            {"inicio": str(raw.loc[pos, DATE].date()),
             "fin": str(raw.loc[pos + length - 1, DATE].date())}
            for pos in selected
        ]
        indices = np.concatenate([np.arange(pos, pos + length) for pos in selected])
        for nombre in (PRICE, VOLUME):
            serie = raw[nombre].copy()
            verdadero = serie.iloc[indices].to_numpy(dtype=float)
            serie.iloc[indices] = np.nan
            candidatos_estimacion = {"LOCF": serie.ffill()}
            for alfa in ALFAS_SENSIBILIDAD:
                estado = serie.ewm(alpha=alfa, adjust=False, ignore_na=True).mean()
                candidatos_estimacion[f"EWMA_{alfa:.2f}"] = serie.where(serie.notna(), estado)
            for metodo, estimacion in candidatos_estimacion.items():
                predicho = estimacion.iloc[indices].to_numpy(dtype=float)
                if not np.isfinite(predicho).all():
                    raise ValueError("No se pudo reconstruir un bloque en la sensibilidad.")
                error = predicho - verdadero
                registros.append({
                    "variable": nombre, "duracion_bloque": length,
                    "n_bloques": len(selected), "n_fechas": len(indices),
                    "metodo": metodo, "MAE": float(np.mean(np.abs(error))),
                    "RMSE": float(np.sqrt(np.mean(error ** 2))),
                    "tipo": "reconstruccion_causal_desarrollo"
                })
    detalles = pd.DataFrame(registros)
    detalles.to_csv(salida, index=False, encoding="utf-8-sig")
    volumen = detalles.loc[(detalles.variable == VOLUME)
                          & (detalles.metodo.str.startswith("EWMA_"))].copy()
    requerido = set(int(k) for k, _ in BLOQUES_SENSIBILIDAD)
    if set(volumen.duracion_bloque) != requerido:
        raise RuntimeError("La sensibilidad no cubre todos los escenarios de duración.")
    tabla = (volumen.pivot(index="metodo", columns="duracion_bloque", values="MAE")
             .reindex(columns=sorted(requerido)))
    if tabla.isna().any().any() or tabla.shape[0] != len(ALFAS_SENSIBILIDAD):
        raise RuntimeError("Faltan métricas para decidir el alpha de EWMA.")
    tabla.columns = [f"MAE_bloque_{int(c)}_dias" for c in tabla.columns]
    tabla["MAE_promedio_equilibrado"] = tabla.mean(axis=1)
    tabla["alpha"] = tabla.index.str.replace("EWMA_", "", regex=False).astype(float)
    tabla = tabla.reset_index().rename(columns={"metodo": "metodo"})
    tabla = tabla.sort_values(["MAE_promedio_equilibrado", "alpha"], kind="stable").reset_index(drop=True)
    elegido = float(tabla.loc[0, "alpha"])
    tabla["seleccionado"] = np.isclose(tabla.alpha, elegido, atol=1e-12)
    tabla.to_csv(seleccion_salida, index=False, encoding="utf-8-sig")

    info = {
        "version": PREPROCESSING_VERSION,
        "alpha_ewma": elegido,
        "variable_calibracion": VOLUME,
        "criterio": CRITERIO_SELECCION_ALPHA,
        "criterio_desempate": "menor_alpha",
        "metrica": "MAE_toneladas",
        "longitudes_bloque": list(sorted(requerido)),
        "alfas_candidatos": list(ALFAS_SENSIBILIDAD),
        "semilla_mascaras": SEED_SENSIBILIDAD,
        "n_filas_desarrollo": n_desarrollo,
        "fecha_desarrollo_fin": str(raw[DATE].max().date()),
        "usa_conjunto_final_prueba": False,
        "resultados_por_alpha": tabla.to_dict(orient="records"),
        "bloques_ocultados": detalle_bloques,
        "archivo_detalle": salida.name,
        "archivo_seleccion": seleccion_salida.name,
    }
    json_salida.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selección automática EWMA: alpha={elegido:.2f} | {CRITERIO_SELECCION_ALPHA}", flush=True)
    print(tabla.to_string(index=False), flush=True)
    print(f"Evidencia: {salida}, {seleccion_salida} y {json_salida}", flush=True)
    return elegido, info


def guardar_metadata_preprocesamiento(df: pd.DataFrame, seleccion: dict) -> None:
    """Manifiesto para verificar Fases 3 y 4 con una misma configuración."""
    if EWMA_ALPHA is None or float(seleccion["alpha_ewma"]) != EWMA_ALPHA:
        raise RuntimeError("El alpha seleccionado no coincide con el alpha utilizado.")
    metadata={"version":PREPROCESSING_VERSION,"alpha_ewma":EWMA_ALPHA,
        "regla_seleccion_alpha": CRITERIO_SELECCION_ALPHA,
        "seleccion_alpha_archivo":"seleccion_alpha_ewma.json",
        "sha256_seleccion_alpha":hashlib.sha256((OUTPUT_DIR/"seleccion_alpha_ewma.json").read_bytes()).hexdigest(),
        "n_filas_calibracion_alpha":int(seleccion["n_filas_desarrollo"]),
        "metodo_precio":"LOCF_solo_predictor", "metodo_volumen":"EWMA_solo_predictor",
        "objetivo":"precio_original_t+1_sin_imputacion",
        "fecha_generacion":datetime.now().isoformat(timespec="seconds"),
        "n_filas":len(df), "n_objetivos_reales":int(df[TARGET_FLAG].sum()),
        "sha256_dataset":hashlib.sha256(OUTPUT_FILE.read_bytes()).hexdigest(),
        "sha256_consolidado":hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest()}
    (OUTPUT_DIR/"protocolo_preprocesamiento.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def ejecutar_con_log(funcion, nombre_fase: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    inicio = datetime.now()
    stdout_original = sys.stdout
    stderr_original = sys.stderr

    with log_file.open("w", encoding="utf-8", buffering=1) as archivo_log:
        sys.stdout = TeeStream(stdout_original, archivo_log)
        sys.stderr = TeeStream(stderr_original, archivo_log)
        estado = "COMPLETADA"
        try:
            print(f"[{inicio:%Y-%m-%d %H:%M:%S}] INICIO {nombre_fase}", flush=True)
            funcion()
        except Exception:
            estado = "ERROR"
            print("\nERROR NO CONTROLADO:", flush=True)
            traceback.print_exc()
            raise
        finally:
            fin = datetime.now()
            duracion = fin - inicio
            print(f"[{fin:%Y-%m-%d %H:%M:%S}] FIN {nombre_fase} | estado={estado} | duración={duracion}", flush=True)
            sys.stdout = stdout_original
            sys.stderr = stderr_original

def main() -> None:
    dataset = pd.read_csv(INPUT_FILE)
    dataset = validar_dataset(dataset)
    consolidado_original = dataset.copy()
    global EWMA_ALPHA
    EWMA_ALPHA, seleccion_ewma = diagnostico_imputacion_con_mascaras(consolidado_original)

    faltantes_precio = int(
        dataset[PRICE].isna().sum()
    )
    faltantes_abastecimiento = int(
        dataset[VOLUME].isna().sum()
    )

    dataset = construir_objetivo(dataset)
    dataset = imputar_series(dataset)
    dataset = construir_caracteristicas_calendario(dataset)
    dataset = construir_rezagos(dataset)
    dataset = construir_caracteristicas_moviles(
        dataset
    )

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    verificar_causalidad_y_rezagos(consolidado_original,dataset)
    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    guardar_metadata_preprocesamiento(dataset, seleccion_ewma)
    generar_evidencia_preprocesamiento(dataset)
    generar_figura_intervalo_faltante(dataset)

    print(
        f"Dataset procesado: {OUTPUT_FILE}"
    )
    print(
        f"Observaciones: {len(dataset)}"
    )
    print(
        f"Periodo de origen: "
        f"{dataset[DATE].min().date()} a "
        f"{dataset[DATE].max().date()}"
    )
    print(
        f"Periodo objetivo: "
        f"{dataset[TARGET_DATE].min().date()} a "
        f"{dataset[TARGET_DATE].max().date()}"
    )
    print(
        f"Faltantes de precio en entradas completados con LOCF: "
        f"{faltantes_precio}"
    )
    print(
        f"Faltantes de abastecimiento en entradas completados con EWMA causal: "
        f"{faltantes_abastecimiento}"
    )
    print(f"EWMA alpha seleccionado con máscaras en desarrollo: {EWMA_ALPHA}")
    print(f"Objetivos de precio t+1 observados: {int(dataset[TARGET_FLAG].sum())}")
    print(f"Objetivos de precio t+1 SIN dato real: {int((dataset[TARGET_FLAG] == 0).sum())}")
    print(
        "Características calendario cíclicas derivadas de fecha_target: "
        f"{CALENDAR_COLUMNS}"
    )
    print(
        f"Rezagos de precio: {PRICE_LAGS}"
    )
    print(
        f"Rezagos de abastecimiento: "
        f"{VOLUME_LAGS}"
    )
    print(
        f"Medias móviles del precio: "
        f"{MOVING_AVERAGE_WINDOWS}"
    )
    print(
        f"Rangos móviles del precio: "
        f"{MOVING_RANGE_WINDOWS}"
    )


if __name__ == "__main__":
    ejecutar_con_log(
        main,
        "FASE 2 - PREPROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS",
        LOG_FILE,
    )