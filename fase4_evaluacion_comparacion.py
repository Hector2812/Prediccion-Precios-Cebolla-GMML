from __future__ import annotations

import json
import sys
import traceback
import math
import os
import re
import shutil
import hashlib
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data procesada" / "dataset_modelado_base.csv"
VALIDATION_DIR = ROOT / "outputs" / "validacion"
MODEL_DIR = VALIDATION_DIR / "modelos"
OUTPUT_DIR = ROOT / "outputs" / "evaluacion_final"
EXPLAIN_DIR = OUTPUT_DIR / "explicabilidad"
FIGURE_DIR = ROOT / "Figuras" / "resultados_discusion"
TABLE_DIR = ROOT / "outputs" / "tablas_resultados"
STATE_FILE = OUTPUT_DIR / "fase4_estado.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "Fase_4_log.txt"

PARTITION_FILE = VALIDATION_DIR / "particion_temporal.json"
PREPROCESSING_MANIFEST = ROOT / "data procesada" / "protocolo_preprocesamiento.json"
PREPROCESSING_VERSION = "Fase2_preprocesamiento_causal_v4_seleccion_alpha"
EWMA_ALPHA: float | None = None
BEST_CONFIG_FILE = VALIDATION_DIR / "mejores_configuraciones.json"
ARTIFACT_FILE = VALIDATION_DIR / "artefactos_modelos.json"

DATE = "fecha"
TARGET_DATE = "fecha_target"
PRICE = "precio_promedio_soles_kg"
VOLUME = "volumen_toneladas"
TARGET = "precio_objetivo_t1"
PRICE_ORIGINAL = "precio_observado_original"
VOLUME_ORIGINAL = "abastecimiento_observado_original"
TARGET_OBSERVED = "objetivo_observado"
PRICE_IMPUTED = "precio_imputado"
VOLUME_IMPUTED = "abastecimiento_imputado"

CALENDAR_COLUMNS = (
    "dia_semana_sin",
    "dia_semana_cos",
    "mes_sin",
    "mes_cos",
    "dia_anio_sin",
    "dia_anio_cos",
)

MODELS = ("SARIMAX", "XGBoost", "LSTM", "TCN", "LSTM-XGBoost")
CONFIG_HISTORICO = "historico"
CONFIG_HISTORICO_ABASTECIMIENTO = "historico_abastecimiento"
CONFIGURATIONS = (CONFIG_HISTORICO, CONFIG_HISTORICO_ABASTECIMIENTO)
CONFIG_LABELS = {
    CONFIG_HISTORICO: "Histórico",
    CONFIG_HISTORICO_ABASTECIMIENTO: "Histórico + abastecimiento",
}
CONFIG_SLUGS = {
    CONFIG_HISTORICO: "historico",
    CONFIG_HISTORICO_ABASTECIMIENTO: "historico_abastecimiento",
}

def etiqueta_configuracion(configuracion: str) -> str:
    return CONFIG_LABELS[configuracion]

def slug_configuracion(configuracion: str) -> str:
    return CONFIG_SLUGS[configuracion]

def clave_modelo(modelo: str, configuracion: str) -> str:
    return f"{modelo}_{slug_configuracion(configuracion)}"

ALPHA = 0.05
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMAPE_EPS = 1e-12
TCN_ARCHITECTURE_ID = "TCN_residual_2conv_causal_v2"

TIMESHAP_NSAMPLES = 2048
TIMESHAP_MAX_INSTANCES = 30
IG_STEPS = 64
IG_MAX_INSTANCES = 60

OMITIR_TIMESHAP = os.getenv("FASE4_OMITIR_TIMESHAP", "0") == "1"


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = self.dropout(hidden[-1])
        return self.output(last_hidden).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Bloque residual TCN con dos convoluciones causales dilatadas."""
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,dilation:int,dropout:float)->None:
        super().__init__()
        self.conv1=CausalConv1d(in_channels,out_channels,kernel_size,dilation); self.relu1=nn.ReLU(); self.dropout1=nn.Dropout(dropout)
        self.conv2=CausalConv1d(out_channels,out_channels,kernel_size,dilation); self.relu2=nn.ReLU(); self.dropout2=nn.Dropout(dropout)
        self.residual=nn.Conv1d(in_channels,out_channels,kernel_size=1) if in_channels!=out_channels else nn.Identity(); self.final_relu=nn.ReLU()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        residual=self.residual(x); y=self.dropout1(self.relu1(self.conv1(x))); y=self.dropout2(self.relu2(self.conv2(y))); return self.final_relu(y+residual)


class TCNRegressor(nn.Module):
    def __init__(self,n_features:int,hidden_channels:int,n_blocks:int,kernel_size:int,dropout:float)->None:
        super().__init__(); blocks=[]; in_channels=n_features
        for level in range(n_blocks):
            blocks.append(TemporalBlock(in_channels,hidden_channels,kernel_size,2**level,dropout)); in_channels=hidden_channels
        self.network=nn.Sequential(*blocks); self.output=nn.Linear(hidden_channels,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        y=self.network(x.transpose(1,2)); return self.output(y[:,:,-1]).squeeze(-1)


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
            print(f"[{fin:%Y-%m-%d %H:%M:%S}] FIN {nombre_fase} | estado={estado} | duración={fin - inicio}", flush=True)
            sys.stdout = stdout_original
            sys.stderr = stderr_original


def imprimir_titulo(texto: str) -> None:
    print("\n" + "=" * 80, flush=True)
    print(texto, flush=True)
    print("=" * 80, flush=True)


def imprimir_subtitulo(texto: str) -> None:
    print("\n" + "-" * 80, flush=True)
    print(texto, flush=True)
    print("-" * 80, flush=True)


def convertir_json(obj):
    if isinstance(obj, dict):
        return {str(k): convertir_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convertir_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def guardar_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(convertir_json(data), file, ensure_ascii=False, indent=2)


def cargar_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolver_ruta_artefacto(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    original = Path(path_value)
    if original.exists():
        return original
    candidate = MODEL_DIR / str(path_value).replace("\\", "/").split("/")[-1]
    if candidate.exists():
        return candidate
    relative = ROOT / original
    if relative.exists():
        return relative
    raise FileNotFoundError(
        f"No se encontró el artefacto '{original.name}' ni en su ruta original ni en {MODEL_DIR}."
    )


def fijar_semilla(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cargar_timeshap_compatible():
    """
    Carga TimeSHAP aplicando una compatibilidad mínima con versiones modernas de SHAP.

    TimeSHAP 1.0.4 intenta importar la clase histórica `Kernel` desde
    `shap.explainers._kernel`, mientras que SHAP moderno expone esa implementación
    como `KernelExplainer`. El alias se crea únicamente en memoria durante esta
    ejecución; no modifica los archivos instalados de SHAP ni TimeSHAP.
    """
    import importlib.util

    if importlib.util.find_spec("shap") is None:
        raise RuntimeError(
            "No se encontró la dependencia 'shap'. Instálela antes de ejecutar la Fase 4."
        )

    if importlib.util.find_spec("timeshap") is None:
        raise RuntimeError(
            "No se encontró la dependencia oficial 'timeshap'. "
            "Instálela con 'pip install timeshap' antes de ejecutar la Fase 4 completa."
        )

    try:
        import shap.explainers._kernel as shap_kernel

        if not hasattr(shap_kernel, "Kernel"):
            if hasattr(shap_kernel, "KernelExplainer"):
                shap_kernel.Kernel = shap_kernel.KernelExplainer
            else:
                raise ImportError(
                    "La versión instalada de SHAP no contiene ni 'Kernel' "
                    "ni 'KernelExplainer'."
                )

        from timeshap.explainer.local_methods import calc_local_report

    except Exception as exc:
        raise RuntimeError(
            f"TimeSHAP está instalado, pero no pudo inicializarse con la versión "
            f"actual de SHAP: {exc}"
        ) from exc

    return calc_local_report


def verificar_dependencias_explicabilidad() -> None:
    import importlib.util

    if importlib.util.find_spec("shap") is None:
        raise RuntimeError(
            "No se encontró la dependencia 'shap'. Instálela antes de ejecutar la Fase 4."
        )

    if not OMITIR_TIMESHAP:
        cargar_timeshap_compatible()


def cargar_dataset() -> pd.DataFrame:
    """Cargar datos y verificar el alpha seleccionado automáticamente en Fase 2."""
    global EWMA_ALPHA
    if not PREPROCESSING_MANIFEST.exists():
        raise FileNotFoundError("Falta protocolo_preprocesamiento.json de la Fase 2.")
    protocol = cargar_json(PREPROCESSING_MANIFEST)
    if protocol.get("version") != PREPROCESSING_VERSION:
        raise ValueError("La versión de la Fase 2 no coincide con la de Fase 4.")
    if protocol.get("regla_seleccion_alpha") != "minimo_MAE_promedio_equilibrado_por_duracion":
        raise ValueError("No se documentó la selección automática de EWMA alpha.")
    evidencia_path = PREPROCESSING_MANIFEST.parent / protocol.get("seleccion_alpha_archivo", "")
    if not evidencia_path.is_file() or hashlib.sha256(evidencia_path.read_bytes()).hexdigest() != protocol.get("sha256_seleccion_alpha"):
        raise ValueError("Falta o cambió la evidencia de selección del alpha EWMA.")
    evidencia = cargar_json(evidencia_path)
    EWMA_ALPHA = float(protocol["alpha_ewma"])
    if not (0 < EWMA_ALPHA <= 1) or not math.isclose(float(evidencia["alpha_ewma"]), EWMA_ALPHA, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Alpha EWMA no es válido o difiere de la selección de Fase 2.")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No se encontró la salida de la Fase 2: {INPUT_FILE}")
    if protocol.get("sha256_dataset") != hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest():
        raise RuntimeError("SHA-256 de Fase 2 diferente: no mezclar ejecuciones.")
    df = pd.read_csv(INPUT_FILE)
    required = [DATE, TARGET_DATE, PRICE, VOLUME, TARGET, PRICE_ORIGINAL,
                VOLUME_ORIGINAL, TARGET_OBSERVED, PRICE_IMPUTED, VOLUME_IMPUTED,
                *CALENDAR_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Se necesita dataset causal de Fase 2; faltan: {missing}")
    for date_col in (DATE, TARGET_DATE):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(DATE).reset_index(drop=True)
    if df[[DATE, TARGET_DATE]].isna().any().any() or df[DATE].duplicated().any():
        raise ValueError("Fechas inválidas o duplicadas.")
    if not df[DATE].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Calendario de origen discontinuo.")
    if not (df[TARGET_DATE] == df[DATE] + pd.Timedelta(days=1)).all():
        raise ValueError("El horizonte no es t+1.")
    mandatory = df[[PRICE, VOLUME, *CALENDAR_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(mandatory).all():
        raise ValueError("Predictores no finitos después del preprocesamiento.")
    if (df[PRICE] <= 0).any() or (df[VOLUME] < 0).any():
        raise ValueError("Predictores fuera de rango.")
    y = df[TARGET].to_numpy(dtype=float)
    observed = np.isfinite(y)
    flags = df[TARGET_OBSERVED].to_numpy(dtype=float)
    if not np.isin(flags, [0, 1]).all() or not np.array_equal(observed, flags.astype(bool)):
        raise ValueError("objetivo_observado no coincide con el precio real.")
    originals = df[PRICE_ORIGINAL].to_numpy(dtype=float)
    if len(y) > 1 and not np.array_equal(y[:-1], originals[1:], equal_nan=True):
        raise ValueError("El objetivo t+1 no es el precio ORIGINAL del día siguiente.")
    if not np.array_equal(df[PRICE].to_numpy(dtype=float),
                          df[PRICE_ORIGINAL].ffill().to_numpy(dtype=float), equal_nan=True):
        raise ValueError("Precio predictor incompatible con LOCF causal.")
    ewma = df[VOLUME_ORIGINAL].ewm(alpha=EWMA_ALPHA, adjust=False, ignore_na=True).mean()
    expected_volume = df[VOLUME_ORIGINAL].where(df[VOLUME_ORIGINAL].notna(), ewma)
    if not np.allclose(df[VOLUME], expected_volume, atol=1e-8, rtol=1e-9, equal_nan=True):
        raise ValueError(f"Volumen predictor incompatible con EWMA causal alfa={EWMA_ALPHA}.")
    return df


def reconstruir_particion(df: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    n_total = int(metadata["n_total"])
    n_implementacion = int(metadata["n_implementacion"])
    n_test = int(metadata["n_prueba"])
    if len(df) != n_total:
        raise ValueError(
            f"El dataset actual tiene {len(df)} filas, pero la Fase 3 registró {n_total}."
        )
    if n_implementacion + n_test != n_total:
        raise ValueError("La partición registrada por la Fase 3 no suma el total de observaciones.")

    implementation_idx = np.arange(0, n_implementacion, dtype=int)
    test_idx = np.arange(n_implementacion, n_total, dtype=int)
    expected = {
        "origen_implementacion_inicio": str(df.loc[implementation_idx[0], DATE].date()),
        "origen_implementacion_fin": str(df.loc[implementation_idx[-1], DATE].date()),
        "objetivo_implementacion_inicio": str(df.loc[implementation_idx[0], TARGET_DATE].date()),
        "objetivo_implementacion_fin": str(df.loc[implementation_idx[-1], TARGET_DATE].date()),
        "origen_prueba_inicio": str(df.loc[test_idx[0], DATE].date()),
        "origen_prueba_fin": str(df.loc[test_idx[-1], DATE].date()),
        "objetivo_prueba_inicio": str(df.loc[test_idx[0], TARGET_DATE].date()),
        "objetivo_prueba_fin": str(df.loc[test_idx[-1], TARGET_DATE].date()),
    }
    for key, value in expected.items():
        if str(metadata.get(key)) != value:
            raise ValueError(
                f"La partición actual no coincide con la Fase 3 en '{key}': "
                f"actual={value}, registrado={metadata.get(key)}."
            )
    state = cargar_json(VALIDATION_DIR / "fase3_estado.json")
    if state.get("estado") != "COMPLETADA":
        raise RuntimeError("La Fase 3 no está marcada como COMPLETADA.")
    if state.get("protocolo_preprocesamiento") != PREPROCESSING_VERSION:
        raise RuntimeError("Protocolo causal incompatible entre Fases 3 y 4.")
    if EWMA_ALPHA is None or not math.isclose(float(state.get("alpha_ewma",-1)), EWMA_ALPHA, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Alpha EWMA usado en Fase 3 no coincide con la evaluación final.")
    if state.get("sha256_protocolo_fase2") != hashlib.sha256(PREPROCESSING_MANIFEST.read_bytes()).hexdigest():
        raise RuntimeError("El protocolo EWMA cambió desde el entrenamiento de Fase 3.")
    digest = hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest()
    if state.get("sha256_dataset_entrada") != digest:
        raise RuntimeError("El CSV de Fase 2 cambió después de entrenar Fase 3: reajuste los modelos.")
    return implementation_idx, test_idx


def obtener_columnas_tabulares(df: pd.DataFrame, configuracion: str) -> list[str]:
    if configuracion not in {CONFIG_HISTORICO, CONFIG_HISTORICO_ABASTECIMIENTO}:
        raise ValueError("La configuración debe ser Histórico o Histórico + abastecimiento.")
    price_columns = [
        column
        for column in df.columns
        if column.startswith("precio_lag_")
        or column.startswith("precio_media_movil_")
        or column.startswith("precio_rango_movil_")
    ]
    calendar_columns = [column for column in CALENDAR_COLUMNS if column in df.columns]
    volume_columns = [
        column for column in df.columns if column.startswith("abastecimiento_lag_")
    ]
    if not price_columns:
        raise ValueError("No se encontraron características temporales del precio.")
    if len(calendar_columns) != len(CALENDAR_COLUMNS):
        raise ValueError(
            "No se encontraron todas las características calendario cíclicas de fecha_target."
        )
    if configuracion == CONFIG_HISTORICO_ABASTECIMIENTO and not volume_columns:
        raise ValueError("No se encontraron rezagos de abastecimiento para Histórico + abastecimiento.")
    base = price_columns + calendar_columns
    return base if configuracion == CONFIG_HISTORICO else base + volume_columns


def codificar_calendario_ciclico(fechas: pd.Series) -> np.ndarray:
    fechas = pd.Series(pd.to_datetime(fechas, errors="coerce"))
    if fechas.isna().any():
        raise ValueError("No se pueden codificar fechas no válidas.")
    dia_semana = fechas.dt.dayofweek.to_numpy(dtype=float)
    mes = fechas.dt.month.to_numpy(dtype=float) - 1.0
    dia_anio = fechas.dt.dayofyear.to_numpy(dtype=float) - 1.0
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


def obtener_nombres_caracteristicas_secuenciales(
    configuracion: str,
    usar_calendario: bool = True,
) -> list[str]:
    if configuracion == CONFIG_HISTORICO:
        base = ["precio"]
    elif configuracion == CONFIG_HISTORICO_ABASTECIMIENTO:
        base = ["precio", "abastecimiento"]
    else:
        raise ValueError("La configuración debe ser Histórico o Histórico + abastecimiento.")
    return [*base, *CALENDAR_COLUMNS] if usar_calendario else base


def preparar_tabular(
    df: pd.DataFrame,
    indices: np.ndarray,
    configuracion: str,
    exigir_objetivo: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = obtener_columnas_tabulares(df, configuracion)
    X = df.loc[indices, columns].to_numpy(dtype=float)
    y = df.loc[indices, TARGET].to_numpy(dtype=float)
    valid = np.isfinite(X).all(axis=1)
    if exigir_objetivo:
        valid &= np.isfinite(y)
    return X[valid], y[valid], indices[valid], columns


def construir_ventanas(
    df: pd.DataFrame,
    indices: np.ndarray,
    window: int,
    configuracion: str,
    usar_calendario: bool = True,
    exigir_objetivo: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if configuracion not in {CONFIG_HISTORICO, CONFIG_HISTORICO_ABASTECIMIENTO}:
        raise ValueError("La configuración debe ser Histórico o Histórico + abastecimiento.")
    price = df[PRICE].to_numpy(dtype=float)
    volume = df[VOLUME].to_numpy(dtype=float)
    target = df[TARGET].to_numpy(dtype=float)
    calendario_historico = (
        codificar_calendario_ciclico(df[DATE]) if usar_calendario else None
    )
    X: list[np.ndarray] = []
    y: list[float] = []
    rows: list[int] = []
    for i in np.asarray(indices, dtype=int):
        start = i - window + 1
        if start < 0:
            continue
        precio_ventana = price[start : i + 1]
        if configuracion == CONFIG_HISTORICO:
            if usar_calendario:
                calendario_ventana = calendario_historico[start : i + 1]
                values = np.column_stack((precio_ventana, calendario_ventana))
            else:
                values = precio_ventana.reshape(-1, 1)
        else:
            abastecimiento_ventana = volume[start : i + 1]
            if usar_calendario:
                calendario_ventana = calendario_historico[start : i + 1]
                values = np.column_stack(
                    (precio_ventana, abastecimiento_ventana, calendario_ventana)
                )
            else:
                values = np.column_stack((precio_ventana, abastecimiento_ventana))
        if not np.isfinite(values).all() or (exigir_objetivo and not np.isfinite(target[i])):
            continue
        X.append(values)
        y.append(float(target[i]))
        rows.append(int(i))
    if not X:
        n_features = len(
            obtener_nombres_caracteristicas_secuenciales(
                configuracion,
                usar_calendario=usar_calendario,
            )
        )
        return (
            np.empty((0, window, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
        )
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(rows, dtype=int),
    )

def escalar_secuencias(
    X: np.ndarray,
    y: np.ndarray,
    x_scaler,
    y_scaler,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, window, n_features = X.shape
    X_scaled = x_scaler.transform(X.reshape(-1, n_features)).reshape(
        n_samples, window, n_features
    )
    y_scaled = y_scaler.transform(np.where(np.isfinite(y), y, 0.0).reshape(-1, 1)).ravel()
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)


def predecir_red(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        predictions = model(tensor).detach().cpu().numpy()
    return predictions


def cargar_lstm(artifacts: dict, configuracion: str) -> tuple[LSTMRegressor, dict, dict]:
    info = artifacts[clave_modelo("LSTM", configuracion)]
    model_path = resolver_ruta_artefacto(info["modelo"])
    scaler_path = resolver_ruta_artefacto(info["escaladores"])
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    params = checkpoint["params"]
    model = LSTMRegressor(
        n_features=int(checkpoint["n_features"]),
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scalers = joblib.load(scaler_path)
    return model, params, scalers


def cargar_tcn(artifacts: dict, configuracion: str) -> tuple[TCNRegressor, dict, dict]:
    info=artifacts[clave_modelo("TCN",configuracion)]
    model_path=resolver_ruta_artefacto(info["modelo"]); scaler_path=resolver_ruta_artefacto(info["escaladores"])
    checkpoint=torch.load(model_path,map_location=DEVICE,weights_only=False)
    arquitectura=checkpoint.get("arquitectura_tcn")
    if arquitectura!=TCN_ARCHITECTURE_ID:
        raise RuntimeError(f"Checkpoint TCN incompatible para {etiqueta_configuracion(configuracion)}: {arquitectura!r}. Ejecute primero la Fase 3 corregida.")
    params=dict(checkpoint["params"]); params["usar_calendario"]=bool(checkpoint.get("usar_calendario",True))
    required=["window","hidden_channels","n_blocks","kernel_size","dropout"]; missing=[k for k in required if k not in params]
    if missing: raise RuntimeError(f"Checkpoint TCN incompleto: faltan {missing}. Vuelva a ejecutar Fase 3.")
    model=TCNRegressor(n_features=int(checkpoint["n_features"]),hidden_channels=int(params["hidden_channels"]),n_blocks=int(params["n_blocks"]),kernel_size=int(params["kernel_size"]),dropout=float(params["dropout"])).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"]); model.eval(); scalers=joblib.load(scaler_path)
    return model,params,scalers


def cargar_xgboost(artifacts: dict, configuracion: str):
    info = artifacts[clave_modelo("XGBoost", configuracion)]
    if info.get("modelo_json"):
        model_path = resolver_ruta_artefacto(info["modelo_json"])
        model = XGBRegressor()
        model.load_model(str(model_path))
        return model, info
    model_path = resolver_ruta_artefacto(info["modelo"])
    return joblib.load(model_path), info


def cargar_sarimax(artifacts: dict, configuracion: str):
    info = artifacts[clave_modelo("SARIMAX", configuracion)]
    model_path = resolver_ruta_artefacto(info["modelo"])
    result = SARIMAXResults.load(str(model_path))
    scaler = None
    if info.get("escalador_exog") is not None:
        scaler_path = resolver_ruta_artefacto(info["escalador_exog"])
        scaler = joblib.load(scaler_path)
    return result, scaler, info


def cargar_corrector_hibrido(artifacts: dict, configuracion: str):
    info = artifacts[clave_modelo("LSTM-XGBoost", configuracion)]
    if info.get("corrector_residual_json"):
        model_path = resolver_ruta_artefacto(info["corrector_residual_json"])
        model = XGBRegressor()
        model.load_model(str(model_path))
        return model, info
    model_path = resolver_ruta_artefacto(info["corrector_residual"])
    return joblib.load(model_path), info


def predecir_baselines(df: pd.DataFrame, test_idx: np.ndarray) -> dict[str, np.ndarray]:
    if "precio_lag_7" not in df.columns:
        raise ValueError("No existe precio_lag_7 para calcular Seasonal Naïve.")
    naive = df.loc[test_idx, PRICE].to_numpy(dtype=float)
    seasonal = df.loc[test_idx, "precio_lag_7"].to_numpy(dtype=float)
    if not np.isfinite(naive).all() or not np.isfinite(seasonal).all():
        raise ValueError("Los baselines contienen valores no finitos en el conjunto de prueba.")
    return {"Naive": naive, "SeasonalNaive_7": seasonal}


def predecir_sarimax_test(
    result,
    scaler,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    configuracion: str,
) -> np.ndarray:
    y_test = df.loc[test_idx, TARGET].to_numpy(dtype=float)
    if configuracion == CONFIG_HISTORICO:
        exog_test = None
    else:
        if scaler is None:
            raise ValueError("SARIMAX - Histórico + abastecimiento no dispone del escalador exógeno guardado en la Fase 3.")
        exog_column = "abastecimiento_lag_1"
        if exog_column not in df.columns:
            raise ValueError("No existe abastecimiento_lag_1 para SARIMAX - Histórico + abastecimiento.")
        raw_exog = df.loc[test_idx, [exog_column]].to_numpy(dtype=float)
        exog_test = scaler.transform(raw_exog)
    predictions: list[float] = []
    current = result
    for position, actual in enumerate(y_test):
        exog_step = None if exog_test is None else exog_test[position : position + 1]
        forecast = current.forecast(steps=1, exog=exog_step)
        predictions.append(float(np.asarray(forecast).ravel()[0]))
        current = current.append(
            endog=np.asarray([actual], dtype=float),
            exog=exog_step,
            refit=False,
        )
    return np.asarray(predictions, dtype=float)


def predecir_xgboost_test(
    model,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    configuracion: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_test, _, rows, columns = preparar_tabular(df, test_idx, configuracion, exigir_objetivo=False)
    if not np.array_equal(rows, test_idx):
        raise ValueError(
            f"XGBoost-{etiqueta_configuracion(configuracion)} no dispone de características válidas para todas las fechas de prueba."
        )
    predictions = np.asarray(model.predict(X_test), dtype=float)
    return predictions, X_test, columns


def predecir_lstm_test(
    model: LSTMRegressor,
    params: dict,
    scalers: dict,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    configuracion: str,
) -> tuple[np.ndarray, dict]:
    X_test, y_test, rows = construir_ventanas(
        df, test_idx, int(params["window"]), configuracion, exigir_objetivo=False
    )
    if not np.array_equal(rows, test_idx):
        raise ValueError(
            f"LSTM-{etiqueta_configuracion(configuracion)} no dispone de secuencias válidas para todas las fechas de prueba."
        )
    X_scaled, _ = escalar_secuencias(
        X_test, y_test, scalers["x_scaler"], scalers["y_scaler"]
    )
    pred_scaled = predecir_red(model, X_scaled)
    predictions = scalers["y_scaler"].inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()
    return predictions, {
        "X_raw": X_test,
        "X_scaled": X_scaled,
        "y": y_test,
        "rows": rows,
    }


def predecir_tcn_test(
    model: TCNRegressor,
    params: dict,
    scalers: dict,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    configuracion: str,
) -> tuple[np.ndarray, dict]:
    X_test, y_test, rows = construir_ventanas(
        df,
        test_idx,
        int(params["window"]),
        configuracion,
        usar_calendario=bool(params.get("usar_calendario", True)),
        exigir_objetivo=False,
    )
    if not np.array_equal(rows, test_idx):
        raise ValueError(
            f"TCN-{etiqueta_configuracion(configuracion)} no dispone de secuencias válidas para todas las fechas de prueba."
        )
    X_scaled, _ = escalar_secuencias(
        X_test, y_test, scalers["x_scaler"], scalers["y_scaler"]
    )
    pred_scaled = predecir_red(model, X_scaled)
    predictions = scalers["y_scaler"].inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()
    return predictions, {
        "X_raw": X_test,
        "X_scaled": X_scaled,
        "y": y_test,
        "rows": rows,
    }


def construir_features_residuales(
    df: pd.DataFrame,
    rows: np.ndarray,
    lstm_predictions: np.ndarray,
    configuracion: str,
) -> tuple[np.ndarray, list[str]]:
    columns = obtener_columnas_tabulares(df, configuracion)
    tabular = df.loc[rows, columns].to_numpy(dtype=float)
    base = np.asarray(lstm_predictions, dtype=float).reshape(-1, 1)
    X = np.column_stack((base, tabular))
    if not np.isfinite(X).all():
        raise ValueError("El corrector residual recibió características no finitas.")
    return X, ["prediccion_lstm"] + columns


def predecir_hibrido_test(
    residual_model,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    configuracion: str,
    lstm_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_res, columns = construir_features_residuales(
        df, test_idx, lstm_predictions, configuracion
    )
    correction = np.asarray(residual_model.predict(X_res), dtype=float)
    hybrid = np.asarray(lstm_predictions, dtype=float) + correction
    return hybrid, X_res, columns


def validar_predicciones(predictions: dict[str, np.ndarray], n_test: int) -> None:
    for name, values in predictions.items():
        arr = np.asarray(values, dtype=float)
        if len(arr) != n_test:
            raise ValueError(
                f"{name} produjo {len(arr)} pronósticos, pero el conjunto final de prueba contiene {n_test}."
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} produjo valores no finitos en el conjunto de prueba.")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    values = np.where(
        denominator <= SMAPE_EPS,
        0.0,
        2.0 * np.abs(y_pred - y_true) / denominator,
    )
    return float(np.mean(values) * 100.0)


def calcular_escala_mase(df: pd.DataFrame, implementation_idx: np.ndarray) -> float:
    """Naive un paso: diferencias sólo entre precios REALES de días consecutivos.

    No se comprime el calendario ni se calculan diferencias a través de días NaN.
    """
    raw = df.loc[implementation_idx, PRICE_ORIGINAL].to_numpy(dtype=float)
    pairs = np.isfinite(raw[1:]) & np.isfinite(raw[:-1])
    differences = np.abs(np.diff(raw)[pairs])
    if not len(differences):
        raise ValueError("MASE: no existen pares consecutivos con precio real.")
    scale = float(np.mean(differences))
    if not np.isfinite(scale) or scale <= SMAPE_EPS:
        raise ValueError("MASE: el denominador es inválido.")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"escala_mase":scale, "pares_consecutivos_reales":len(differences),
                   "tipo":"naive no estacional 1 día, desarrollo, sin imputar"}]).to_csv(
        TABLE_DIR/"Tabla_4_escala_MASE.csv",index=False,encoding="utf-8-sig")
    return scale


def calcular_metricas_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_scale: float,
) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "smape": smape(y_true, y_pred),
        "mae": mae,
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mase": float(mae / mase_scale),
        "r2": float(r2_score(y_true, y_pred)),
    }


def construir_tabla_metricas(prediction_table: pd.DataFrame, mase_scale: float) -> pd.DataFrame:
    y_true = prediction_table["real"].to_numpy(dtype=float)
    valid_real = np.isfinite(y_true)
    if not valid_real.any():
        raise ValueError("No existen objetivos reales en la prueba.")
    rows = []
    pairs = [(name, "Referencia", name) for name in ("Naive", "SeasonalNaive_7")]
    pairs += [(model, etiqueta_configuracion(cfg), clave_modelo(model,cfg))
              for cfg in CONFIGURATIONS for model in MODELS]
    for model,config,column in pairs:
        yp = prediction_table[column].to_numpy(dtype=float)
        if not np.isfinite(yp[valid_real]).all():
            raise ValueError(f"Pronóstico inválido en una fecha evaluable: {column}")
        metricas = calcular_metricas_test(y_true[valid_real],yp[valid_real],mase_scale)
        rows.append({"modelo":model,"configuracion":config, **metricas,
                     "n":int(valid_real.sum()),"objetivos_ausentes":int((~valid_real).sum())})
    return pd.DataFrame(rows)


def construir_comparacion_historico_abastecimiento(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODELS:
        historico = metrics[(metrics["modelo"] == model) & (metrics["configuracion"] == etiqueta_configuracion(CONFIG_HISTORICO))].iloc[0]
        abastecimiento = metrics[(metrics["modelo"] == model) & (metrics["configuracion"] == etiqueta_configuracion(CONFIG_HISTORICO_ABASTECIMIENTO))].iloc[0]
        row = {"modelo": model}
        for metric in ("smape", "mae", "rmse", "mase", "r2"):
            row[f"{metric}_historico"] = float(historico[metric])
            row[f"{metric}_historico_abastecimiento"] = float(abastecimiento[metric])
            row[f"delta_{metric}_abastecimiento_menos_historico"] = float(abastecimiento[metric] - historico[metric])
        rows.append(row)
    return pd.DataFrame(rows)

def calcular_nlags_hac(n: int) -> int:
    if n <= 1:
        return 0
    value = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(0, min(value, n - 1))


def contraste_diebold_mariano_hac(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
) -> dict[str, float | int | str]:
    """DM bilateral con pérdidas cuadráticas y Bartlett HAC.

    Para calendarios sin faltantes se conserva el cálculo statsmodels original.
    Si hay objetivos NaN se preserva la distancia de días al estimar HAC;
    no se conectan artificialmente extremos de lagunas de datos.
    """
    y_true, pred_a, pred_b = [np.asarray(x,dtype=float) for x in (y_true,pred_a,pred_b)]
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("DM requiere pronósticos emparejados.")
    valid = np.isfinite(y_true)
    if valid.sum() < 3 or not np.isfinite(pred_a[valid]).all() or not np.isfinite(pred_b[valid]).all():
        raise ValueError("DM requiere al menos tres pares evaluables finitos.")
    diff = np.full(len(y_true), np.nan, dtype=float)
    diff[valid] = (y_true[valid]-pred_a[valid])**2 - (y_true[valid]-pred_b[valid])**2
    n=int(valid.sum()); lag=calcular_nlags_hac(n)
    if np.allclose(pred_a[valid], pred_b[valid], rtol=1e-9, atol=1e-9):
        return {"dm_estadistico":0.0,"p_valor":1.0,
            "diferencial_perdida_medio_A_menos_B":0.0,
            "menor_perdida_cuadratica_media":"Igual","n":n,
            "hac_nlags":lag,"pares_temporales_ausentes":int((~valid).sum()),
            "equivalencia_numerica":True}
    mean_d=float(np.nanmean(diff))
    if valid.all():
        regression=sm.OLS(diff,np.ones((n,1))).fit()
        variance_mean=float(cov_hac(regression,nlags=lag,use_correction=True)[0,0])
    else:
        centered=np.where(valid,diff-mean_d,0.)
        longrun=float(np.dot(centered,centered))
        for j in range(1,lag+1):
            pairs=valid[j:] & valid[:-j]
            longrun += 2*(1-j/(lag+1))*float(np.dot(centered[j:][pairs],centered[:-j][pairs]))
        variance_mean=max(0.,longrun / max(n-1,1) / n)
    if variance_mean <= 0 or not np.isfinite(variance_mean):
        if np.nanmax(diff)-np.nanmin(diff) <= 1e-10:
            statistic=0.0 if abs(mean_d)<=SMAPE_EPS else math.copysign(math.inf,mean_d)
            p_value=1.0 if statistic==0.0 else 0.0
        else:
            raise ValueError("DM: varianza HAC degenerada.")
    else:
        statistic=float(mean_d/math.sqrt(variance_mean))
        p_value=float(2*norm.sf(abs(statistic)))
    menor = "A" if mean_d<0 else "B" if mean_d>0 else "Igual"
    return {"dm_estadistico":statistic,"p_valor":p_value,
            "diferencial_perdida_medio_A_menos_B":mean_d,
            "menor_perdida_cuadratica_media":menor,"n":n,"hac_nlags":lag,
            "pares_temporales_ausentes":int((~valid).sum()),
            "equivalencia_numerica":False}


def aplicar_holm(table: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    result = table.copy()
    reject, p_adjusted, _, _ = multipletests(
        result["p_valor"].to_numpy(dtype=float),
        alpha=alpha,
        method="holm",
    )
    result["p_valor_holm"] = p_adjusted
    result["rechaza_h0_holm"] = reject
    result["alpha"] = alpha
    return result


def comparar_modelos_dm(
    prediction_table: pd.DataFrame,
    configuracion: str,
) -> pd.DataFrame:
    y_true = prediction_table["real"].to_numpy(dtype=float)
    rows: list[dict] = []
    for model_a, model_b in combinations(MODELS, 2):
        pred_a = prediction_table[clave_modelo(model_a, configuracion)].to_numpy(dtype=float)
        pred_b = prediction_table[clave_modelo(model_b, configuracion)].to_numpy(dtype=float)
        dm = contraste_diebold_mariano_hac(y_true, pred_a, pred_b)
        winner = model_a if dm["menor_perdida_cuadratica_media"] == "A" else model_b if dm["menor_perdida_cuadratica_media"] == "B" else "Igual"
        rows.append({
            "familia": f"modelos_{slug_configuracion(configuracion)}",
            "configuracion": etiqueta_configuracion(configuracion),
            "modelo_A": model_a, "modelo_B": model_b,
            "modelo_menor_perdida_cuadratica_media": winner, **dm,
        })
    return aplicar_holm(pd.DataFrame(rows))


def comparar_historico_abastecimiento_dm(prediction_table: pd.DataFrame) -> pd.DataFrame:
    y_true = prediction_table["real"].to_numpy(dtype=float)
    rows: list[dict] = []
    for model in MODELS:
        pred_historico = prediction_table[clave_modelo(model, CONFIG_HISTORICO)].to_numpy(dtype=float)
        pred_abastecimiento = prediction_table[clave_modelo(model, CONFIG_HISTORICO_ABASTECIMIENTO)].to_numpy(dtype=float)
        dm = contraste_diebold_mariano_hac(y_true, pred_historico, pred_abastecimiento)
        winner = (
            etiqueta_configuracion(CONFIG_HISTORICO)
            if dm["menor_perdida_cuadratica_media"] == "A"
            else etiqueta_configuracion(CONFIG_HISTORICO_ABASTECIMIENTO)
            if dm["menor_perdida_cuadratica_media"] == "B"
            else "Igual"
        )
        rows.append({
            "familia": "historico_vs_historico_abastecimiento",
            "modelo": model,
            "configuracion_A": etiqueta_configuracion(CONFIG_HISTORICO),
            "configuracion_B": etiqueta_configuracion(CONFIG_HISTORICO_ABASTECIMIENTO),
            "configuracion_menor_perdida_cuadratica_media": winner,
            **dm,
        })
    return aplicar_holm(pd.DataFrame(rows))

def comparar_baselines_dm(prediction_table: pd.DataFrame, configuracion: str) -> pd.DataFrame:
    """Contrastes modelo vs Naive y Seasonal Naive (10 por configuración).

    Esta familia Holm está definida antes de evaluar los resultados.
    """
    real = prediction_table["real"].to_numpy(dtype=float)
    rows=[]
    for baseline in ("Naive","SeasonalNaive_7"):
        base=prediction_table[baseline].to_numpy(dtype=float)
        for model in MODELS:
            pred=prediction_table[clave_modelo(model,configuracion)].to_numpy(dtype=float)
            dm=contraste_diebold_mariano_hac(real,pred,base)
            rows.append({"modelo_A":model,"modelo_B":baseline,
                         "familia":f"modelos_vs_baselines_{slug_configuracion(configuracion)}",
                         "configuracion":etiqueta_configuracion(configuracion),**dm})
    return aplicar_holm(pd.DataFrame(rows))


def seleccionar_posiciones_representativas(n: int, max_instances: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=int)
    if n <= max_instances:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_instances, dtype=int))


def explicar_xgboost_treeshap(
    model,
    X: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("TreeSHAP devolvió una forma incompatible con las características del modelo.")
    result = pd.DataFrame(
        {
            "predictor": feature_names,
            "shap_medio": np.mean(values, axis=0),
            "shap_abs_medio": np.mean(np.abs(values), axis=0),
        }
    ).sort_values("shap_abs_medio", ascending=False)
    result["ranking"] = np.arange(1, len(result) + 1)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def construir_funcion_timeshap_lstm(model: LSTMRegressor, y_scaler):
    def f(x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        pred_scaled = predecir_red(model, array)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        return np.asarray(pred, dtype=float)

    return f


def explicar_lstm_timeshap(
    model: LSTMRegressor,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_names: list[str],
    y_scaler,
    target_dates: np.ndarray,
    prefix: Path,
) -> dict[str, pd.DataFrame]:
    cargar_timeshap_compatible()
    from timeshap.explainer import local_event, local_feat
    from timeshap.explainer.local_methods import validate_local_input

    positions = seleccionar_posiciones_representativas(
        len(X_test_scaled), TIMESHAP_MAX_INSTANCES
    )
    baseline = np.mean(
        X_train_scaled.reshape(-1, X_train_scaled.shape[2]), axis=0
    ).reshape(1, -1)
    f = construir_funcion_timeshap_lstm(model, y_scaler)
    event_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    event_dict = {"rs": SEED, "nsamples": TIMESHAP_NSAMPLES}
    feature_dict = {
        "rs": SEED,
        "nsamples": TIMESHAP_NSAMPLES,
        "feature_names": feature_names,
    }
    print(
        f"TimeSHAP: {len(positions)} secuencias de {X_test_scaled.shape[1]} días; "
        "se explica la ventana completa, sin poda temporal.",
        flush=True,
    )
    for step, position in enumerate(positions, start=1):
        start_instance = perf_counter()
        target_date = pd.Timestamp(target_dates[position])
        print(
            f"  [{step}/{len(positions)}] Fecha objetivo {target_date.date()}: calculando...",
            flush=True,
        )
        sequence = X_test_scaled[position : position + 1].astype(float)

        validate_local_input(
            f=f,
            data=sequence,
            pruning_dict=None,
            event_dict=event_dict,
            feature_dict=feature_dict,
            cell_dict=None,
            baseline=baseline,
            model_features=list(range(len(feature_names))),
            entity_col=None,
            entity_uuid=None,
            time_col=None,
        )

        event_data = local_event(
            f=f,
            data=sequence,
            event_dict=event_dict,
            entity_uuid=None,
            entity_col=None,
            baseline=baseline,
            pruned_idx=0,
        )
        feature_data = local_feat(
            f=f,
            data=sequence,
            feature_dict=feature_dict,
            entity_uuid=None,
            entity_col=None,
            baseline=baseline,
            pruned_idx=0,
        )
        event_data = event_data.copy()
        feature_data = feature_data.copy()
        event_data["fecha_target"] = pd.Timestamp(target_dates[position])
        feature_data["fecha_target"] = pd.Timestamp(target_dates[position])
        event_frames.append(event_data)
        feature_frames.append(feature_data)
        print(
            f"      Completado en {perf_counter() - start_instance:.1f} s.",
            flush=True,
        )
    events_local = pd.concat(event_frames, ignore_index=True)
    features_local = pd.concat(feature_frames, ignore_index=True)
    events_global = (
        events_local.groupby("Feature", as_index=False)["Shapley Value"]
        .agg(atribucion_media="mean", atribucion_abs_media=lambda x: float(np.mean(np.abs(x))))
        .sort_values("atribucion_abs_media", ascending=False)
        .reset_index(drop=True)
    )
    features_global = (
        features_local.groupby("Feature", as_index=False)["Shapley Value"]
        .agg(atribucion_media="mean", atribucion_abs_media=lambda x: float(np.mean(np.abs(x))))
        .sort_values("atribucion_abs_media", ascending=False)
        .reset_index(drop=True)
    )
    events_global["ranking"] = np.arange(1, len(events_global) + 1)
    features_global["ranking"] = np.arange(1, len(features_global) + 1)
    events_local.to_csv(
        prefix.with_name(prefix.name + "_eventos_local.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    features_local.to_csv(
        prefix.with_name(prefix.name + "_caracteristicas_local.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    events_global.to_csv(
        prefix.with_name(prefix.name + "_eventos_global.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    features_global.to_csv(
        prefix.with_name(prefix.name + "_caracteristicas_global.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"TimeSHAP completado: {len(positions)} secuencias; 4 CSV guardados.", flush=True)
    return {"eventos": events_global, "caracteristicas": features_global}


def integrated_gradients(
    model: nn.Module,
    sample: np.ndarray,
    baseline: np.ndarray,
    output_scale: float,
    steps: int = IG_STEPS,
) -> np.ndarray:
    if steps < 2:
        raise ValueError("Integrated Gradients requiere al menos dos pasos de integración.")
    x = torch.from_numpy(sample.astype(np.float32)).to(DEVICE)
    b = torch.from_numpy(baseline.astype(np.float32)).to(DEVICE)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=DEVICE)
    gradients: list[torch.Tensor] = []
    model.eval()
    for alpha in alphas:
        interpolated = (b + alpha * (x - b)).clone().detach().requires_grad_(True)
        output = model(interpolated.unsqueeze(0)).squeeze()
        model.zero_grad(set_to_none=True)
        if interpolated.grad is not None:
            interpolated.grad.zero_()
        output.backward()
        gradients.append(interpolated.grad.detach().clone())
    stacked = torch.stack(gradients, dim=0)
    trapezoids = (stacked[:-1] + stacked[1:]) / 2.0
    avg_grad = trapezoids.mean(dim=0)
    attribution = (x - b) * avg_grad * float(output_scale)
    return attribution.detach().cpu().numpy()


def explicar_tcn_integrated_gradients(
    model: TCNRegressor,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_names: list[str],
    y_scaler,
    target_dates: np.ndarray,
    prefix: Path,
) -> dict[str, pd.DataFrame]:
    positions = seleccionar_posiciones_representativas(
        len(X_test_scaled), IG_MAX_INSTANCES
    )
    baseline = np.mean(X_train_scaled, axis=0).astype(np.float32)
    output_scale = float(1.0 / y_scaler.scale_[0])
    local_rows: list[dict] = []
    for position in positions:
        attr = integrated_gradients(
            model,
            X_test_scaled[position],
            baseline,
            output_scale,
            steps=IG_STEPS,
        )
        window = attr.shape[0]
        for time_pos in range(window):
            lag_from_origin = window - 1 - time_pos
            for feature_index, feature_name in enumerate(feature_names):
                value = float(attr[time_pos, feature_index])
                local_rows.append(
                    {
                        "fecha_target": pd.Timestamp(target_dates[position]),
                        "posicion_ventana": int(time_pos),
                        "rezago_desde_origen": int(lag_from_origin),
                        "variable": feature_name,
                        "atribucion": value,
                        "atribucion_abs": abs(value),
                    }
                )
    local = pd.DataFrame(local_rows)
    temporal = (
        local.groupby("rezago_desde_origen", as_index=False)
        .agg(
            atribucion_media=("atribucion", "mean"),
            atribucion_abs_media=("atribucion_abs", "mean"),
        )
        .sort_values("atribucion_abs_media", ascending=False)
        .reset_index(drop=True)
    )
    features = (
        local.groupby("variable", as_index=False)
        .agg(
            atribucion_media=("atribucion", "mean"),
            atribucion_abs_media=("atribucion_abs", "mean"),
        )
        .sort_values("atribucion_abs_media", ascending=False)
        .reset_index(drop=True)
    )
    temporal["ranking"] = np.arange(1, len(temporal) + 1)
    features["ranking"] = np.arange(1, len(features) + 1)
    local.to_csv(
        prefix.with_name(prefix.name + "_local.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    temporal.to_csv(
        prefix.with_name(prefix.name + "_temporal.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    features.to_csv(
        prefix.with_name(prefix.name + "_variables.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return {"temporal": temporal, "variables": features}


def explicar_sarimax(result, configuracion: str, output_path: Path) -> pd.DataFrame:
    names = list(result.param_names)
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    table = pd.DataFrame(
        {
            "parametro": names,
            "coeficiente": params,
            "error_estandar": bse,
            "p_valor": pvalues,
        }
    )
    table["configuracion"] = etiqueta_configuracion(configuracion)
    table["predictor_exogeno_asociado"] = ""
    if configuracion == CONFIG_HISTORICO_ABASTECIMIENTO:
        exog_names = [name for name in names if name.lower().startswith("x")]
        if len(exog_names) == 1:
            table.loc[
                table["parametro"] == exog_names[0], "predictor_exogeno_asociado"
            ] = "abastecimiento_lag_1_estandarizado"
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return table


def registrar_componente_lstm_hibrido(configuracion: str) -> None:
    source_prefix = EXPLAIN_DIR / f"lstm_{slug_configuracion(configuracion)}_timeshap"
    target_prefix = EXPLAIN_DIR / f"hibrido_lstm_{slug_configuracion(configuracion)}_timeshap"
    suffixes = (
        "_eventos_local.csv",
        "_caracteristicas_local.csv",
        "_eventos_global.csv",
        "_caracteristicas_global.csv",
    )
    for suffix in suffixes:
        source = source_prefix.with_name(source_prefix.name + suffix)
        target = target_prefix.with_name(target_prefix.name + suffix)
        if source.exists():
            shutil.copyfile(source, target)


def obtener_versiones_librerias() -> dict[str, str | None]:
    import importlib.metadata as metadata

    packages = [
        "numpy",
        "pandas",
        "torch",
        "xgboost",
        "scikit-learn",
        "statsmodels",
        "scipy",
        "joblib",
        "shap",
        "timeshap",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _guardar_figura(fig, nombre: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURE_DIR / nombre
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura generada: {ruta}", flush=True)
    return ruta


def generar_tablas_y_figuras_resultados(
    table: pd.DataFrame, metrics: pd.DataFrame, comparison: pd.DataFrame,
    dm_h: pd.DataFrame, dm_a: pd.DataFrame, dm_pair: pd.DataFrame,
    dm_baselines: pd.DataFrame,
) -> None:
    """Gráficos del 20 % de prueba, UNA SOLA VEZ después de fijar los modelos."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_DIR / "Tabla_4_metricas_finales_todos_los_modelos.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLE_DIR / "Tabla_5_aporte_abastecimiento.csv", index=False, encoding="utf-8-sig")
    dm_h.to_csv(TABLE_DIR / "Tabla_6_DM_Holm_historico.csv", index=False, encoding="utf-8-sig")
    dm_a.to_csv(TABLE_DIR / "Tabla_6_DM_Holm_abastecimiento.csv", index=False, encoding="utf-8-sig")
    dm_pair.to_csv(TABLE_DIR / "Tabla_6_DM_Holm_abastecimiento_vs_historico.csv", index=False, encoding="utf-8-sig")
    dm_baselines.to_csv(TABLE_DIR / "Tabla_6b_DM_Holm_modelos_vs_baselines.csv",index=False,encoding="utf-8-sig")
    fecha = pd.to_datetime(table["fecha_target"])
    observed = np.isfinite(table["real"])
    y = table.loc[observed, "real"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    xs = np.arange(len(MODELS)); w = 0.38
    for side, cfg in enumerate(CONFIGURATIONS):
        temp = metrics[metrics["configuracion"].eq(etiqueta_configuracion(cfg))].set_index("modelo")
        values = [float(temp.loc[m, "smape"]) for m in MODELS]
        ax.bar(xs + (-w/2 if side==0 else w/2), values, width=w,
               label=etiqueta_configuracion(cfg), alpha=.85)
    for name, style in (("Naive", "--"),("SeasonalNaive_7", ":")):
        value = float(metrics.loc[metrics.modelo.eq(name), "smape"].iloc[0])
        ax.axhline(value, linestyle=style, color="black" if name=="Naive" else "gray",
                   label=f"{name} ({value:.2f} %)", lw=1.2)
    ax.set_xticks(xs,MODELS,rotation=12,ha="right")
    ax.set(ylabel="sMAPE (%) - menor es mejor",title="Pronóstico t+1: desempeño en la prueba final")
    ax.legend(fontsize=8,ncol=2);ax.grid(axis="y",alpha=.16);fig.tight_layout()
    _guardar_figura(fig,"Figura_6_1_comparacion_smape_final.png")

    comparados = [
        ("Naive", "Naive", "#696969", "--"),
        ("LSTM - Hist. + abastecimiento", clave_modelo("LSTM", CONFIG_HISTORICO_ABASTECIMIENTO), "#1c78a8", "-"),
        ("LSTM-XGBoost - Hist. + abastecimiento", clave_modelo("LSTM-XGBoost", CONFIG_HISTORICO_ABASTECIMIENTO), "#a0642d", "-"),
    ]
    sarimax_h = clave_modelo("SARIMAX", CONFIG_HISTORICO)
    if np.allclose(table["Naive"].to_numpy(dtype=float),
                   table[sarimax_h].to_numpy(dtype=float), rtol=1e-9, atol=1e-9):
        comparados[0] = ("Naive / SARIMAX Histórico (coinciden)", "Naive", "#696969", "--")
    else:
        comparados.append(("SARIMAX - Histórico", sarimax_h, "#79599a", ":"))
    ordered = metrics.sort_values("smape")
    best = ordered[~ordered["modelo"].isin(["Naive", "SeasonalNaive_7"])].iloc[0]
    cfg = CONFIG_HISTORICO if best["configuracion"] == "Histórico" else CONFIG_HISTORICO_ABASTECIMIENTO
    best_col = clave_modelo(str(best["modelo"]), cfg)
    zoom_n = min(180, len(table))
    fechas_zoom = fecha.iloc[-zoom_n:]
    fecha_inicio, fecha_fin = fechas_zoom.iloc[0], fechas_zoom.iloc[-1]
    fig, axs = plt.subplots(2, 1, figsize=(12.2, 7.4), sharey=True)
    ymin = float(min(table["real"].min(), min(table[col].min() for _, col, _, _ in comparados)))
    ymax = float(max(table["real"].max(), max(table[col].max() for _, col, _, _ in comparados)))
    pad = max((ymax-ymin)*0.05, 0.08)
    for ax, tramo, titulo in (
        (axs[0], slice(None), "(a) Periodo completo de prueba"),
        (axs[1], slice(len(table)-zoom_n, None),
         f"(b) Ampliación: {fecha_inicio:%d/%m/%Y} a {fecha_fin:%d/%m/%Y}"),
    ):
        ax.plot(fecha.iloc[tramo], table["real"].iloc[tramo], color="#171717",
                lw=1.5, label="Precio observado", zorder=5)
        for etiqueta, col, color, estilo in comparados:
            ax.plot(fecha.iloc[tramo], table[col].iloc[tramo],
                    label=etiqueta, color=color, linestyle=estilo, linewidth=1.0, alpha=.90)
        ax.set_ylim(ymin-pad, ymax+pad)
        ax.set(ylabel="Precio (S/kg)", title=titulo)
        ax.grid(alpha=0.15)
    axs[0].axvspan(fecha_inicio, fecha_fin, facecolor="#dbeafe", edgecolor="#5484ac",
                   linewidth=1.0, alpha=0.60, label="Intervalo ampliado")
    from matplotlib.ticker import MaxNLocator
    for ax in axs:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axs[0].legend(loc="upper left", fontsize=8, ncol=2)
    axs[1].set_xlabel("Fecha objetivo del pronóstico (t+1)")
    fig.suptitle("Precio observado y pronósticos diarios - evaluación final", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    _guardar_figura(fig, "Figura_6_2_reales_vs_pronosticados.png")
    fig,ax=plt.subplots(figsize=(9,4.6))
    values=comparison.set_index("modelo").loc[list(MODELS),"delta_smape_abastecimiento_menos_historico"]
    pos=np.arange(len(values));ax.barh(pos,values,color=["#c46443" if x>0 else "#327b71" for x in values])
    ax.set_yticks(pos,values.index);ax.axvline(0,color="black",lw=.8)
    ax.set_xlabel("Δ sMAPE (puntos porcentuales): con abastecimiento menos Histórico")
    ax.set_title("Cambio de error al incorporar abastecimiento (negativo: mejora)")
    ax.grid(axis="x",alpha=.16);fig.tight_layout()
    _guardar_figura(fig,"Figura_6_3_aporte_abastecimiento.png")
    fig,axs=plt.subplots(1,2,figsize=(11.8,5.1),constrained_layout=True)
    image=None
    for ax,comp,cfg_name in zip(axs,[dm_h,dm_a],["Histórico","Histórico + abastecimiento"]):
        matrix=np.full((len(MODELS),len(MODELS)),np.nan)
        marks=np.full((len(MODELS),len(MODELS)),False)
        for _,r in comp.iterrows():
            a=MODELS.index(r["modelo_A"]);b=MODELS.index(r["modelo_B"])
            val=float(r["p_valor_holm"]);sig=bool(r["rechaza_h0_holm"])
            matrix[a,b]=matrix[b,a]=val;marks[a,b]=marks[b,a]=sig
        image=ax.imshow(-np.log10(np.clip(np.nan_to_num(matrix,nan=1.),1e-12,1.)),
                        vmin=0,vmax=5,cmap="Blues")
        ax.set_xticks(range(len(MODELS)),MODELS,rotation=40,ha="right",fontsize=8)
        ax.set_yticks(range(len(MODELS)),MODELS,fontsize=8)
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                if i!=j:
                    ax.text(j,i,"*" if marks[i,j] else "·",va="center",ha="center",
                            color="white" if matrix[i,j]<1e-3 else "black")
        ax.set_title(cfg_name)
    fig.colorbar(image,ax=axs,shrink=.75,label="-log10(p Holm); * significativo, α=0.05")
    _guardar_figura(fig,"Figura_6_4_DM_Holm_matriz.png")

    fig, axs = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True, sharey=True)
    maxima = []
    for cfg in CONFIGURATIONS:
        sub = dm_baselines[(dm_baselines["configuracion"] == etiqueta_configuracion(cfg)) &
                           (dm_baselines["modelo_B"] == "Naive")].set_index("modelo_A").reindex(MODELS)
        if sub["p_valor_holm"].isna().any():
            raise ValueError("Faltan contrastes frente a Naive en uno de los conjuntos.")
        maxima.extend(sub["diferencial_perdida_medio_A_menos_B"].abs().to_list())
    bound = max(max(maxima)*1.16, 1e-7)
    for ax, cfg in zip(axs, CONFIGURATIONS):
        sub = dm_baselines[(dm_baselines["configuracion"] == etiqueta_configuracion(cfg)) &
                           (dm_baselines["modelo_B"] == "Naive")].set_index("modelo_A").reindex(MODELS)
        valores = sub["diferencial_perdida_medio_A_menos_B"].to_numpy(dtype=float)
        sig = sub["rechaza_h0_holm"].astype(bool).to_numpy()
        pvals = sub["p_valor_holm"].to_numpy(dtype=float)
        colores = ["#b45346" if d > 0 else "#287e68" for d in valores]
        barras = ax.barh(np.arange(len(MODELS)), valores, color=colores, height=.64)
        for bar, sig_item in zip(barras, sig):
            bar.set_alpha(1.0 if sig_item else 0.40)
        ax.axvline(0, color="#202020", lw=1.0)
        ax.set_yticks(np.arange(len(MODELS)), MODELS)
        ax.set_xlim(-bound, bound)
        ax.set_ylabel(etiqueta_configuracion(cfg))
        ax.grid(axis="x", alpha=.15)
        for i, (pval, significativo) in enumerate(zip(pvals, sig)):
            texto = f"p Holm = {pval:.3g}" + (" *" if significativo else "")
            ax.text(1.02, i, texto, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8)
    axs[0].invert_yaxis()
    axs[-1].set_xlabel("Diferencia media de pérdida cuadrática: modelo menos Naive (S/kg)^2")
    fig.suptitle("Contrastes Diebold-Mariano frente a Naive con ajuste de Holm", fontsize=12)
    fig.text(.12, .015,
             "Negativo: favorece al modelo. Positivo: favorece a Naive. * p ajustado < 0,05. "
             "Sin asterisco no demuestra equivalencia.", fontsize=9)
    fig.subplots_adjust(left=.18, right=.79, top=.90, bottom=.11, hspace=.24)
    _guardar_figura(fig, "Figura_6_4b_DM_Holm_vs_Naive.png")
    cols=["Naive"]+[clave_modelo(m,CONFIG_HISTORICO) for m in MODELS]+[
        clave_modelo(m,CONFIG_HISTORICO_ABASTECIMIENTO) for m in MODELS]
    errs=[(table.loc[observed,c].to_numpy(dtype=float)-y) for c in cols]
    fig,ax=plt.subplots(figsize=(12,6.5));ax.boxplot(errs,showfliers=False,vert=False)
    ax.set_yticks(range(1,len(cols)+1),cols,fontsize=8)
    ax.axvline(0,color="black",lw=.8)
    ax.set(xlabel="Error firmado = pronóstico - observado (S/kg)",
           title="Distribución de errores en la prueba final (sin valores atípicos visualizados)")
    ax.grid(axis="x",alpha=.17);fig.tight_layout()
    _guardar_figura(fig,"Figura_6_5_errores_cajas.png")
    records=[]
    for column in cols:
        part=pd.DataFrame({"anio":fecha[observed].dt.year.to_numpy(),
            "error_absoluto":np.abs(table.loc[observed,column].to_numpy(dtype=float)-y)})
        for year,grp in part.groupby("anio"):
            records.append({"modelo_columna":column,"anio":int(year),
                            "MAE_S_kg":float(grp.error_absoluto.mean()),"n":len(grp)})
    by_year=pd.DataFrame(records)
    by_year.to_csv(TABLE_DIR/"Tabla_7_errores_por_anio.csv",index=False,encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(10.5,4.6))
    for col in ["Naive",best_col,clave_modelo("LSTM-XGBoost",cfg)]:
        z=by_year.loc[by_year.modelo_columna.eq(col)]
        ax.plot(z.anio,z.MAE_S_kg,marker="o",label=col)
    ax.set(xlabel="Año objetivo",ylabel="MAE (S/kg)",title="Errores por año durante la prueba final")
    ax.legend(fontsize=8);ax.grid(alpha=.17);fig.tight_layout()
    _guardar_figura(fig,"Figura_6_5b_errores_por_anio.png")
    print("Tablas de resultados, DM-Holm y figuras de desempeño generadas.",flush=True)


def _figura_atribuciones_globales(table:pd.DataFrame,nombre:str,ruta:str,
                                  name_col:str,value_col:str="atribucion_abs_media") -> None:
    if table.empty or name_col not in table or value_col not in table:
        raise ValueError(f"Faltan columnas para figura explicativa: {nombre}")
    top=table.nlargest(min(15,len(table)),value_col).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8.8,max(3.5,.32*len(top)+1.5)))
    ax.barh(top[name_col].astype(str),top[value_col])
    ax.set(xlabel="Media del valor absoluto de la atribución (S/kg)",title=nombre)
    ax.grid(axis="x",alpha=.17);fig.tight_layout();_guardar_figura(fig,ruta)


def _figura_local_barras(df:pd.DataFrame,nombre:str,ruta:str,
                         label_col:str="Feature",value_col:str="Shapley Value") -> None:
    vals=df.groupby(label_col)[value_col].sum().sort_values(key=lambda s:s.abs(),ascending=False).head(12)
    vals=vals.iloc[::-1]
    fig,ax=plt.subplots(figsize=(8.5,max(3.5,len(vals)*.32+1.5)))
    ax.barh(vals.index.astype(str),vals,color=["#326f91" if v>=0 else "#c45b45" for v in vals])
    ax.axvline(0,color="black",lw=.7)
    ax.set(xlabel="Atribución firmada a este pronóstico (S/kg)",title=nombre)
    ax.grid(axis="x",alpha=.15);fig.tight_layout();_guardar_figura(fig,ruta)




def _rotulo_predictor(nombre: str) -> str:
    """Etiquetas compactas sin cambiar la identidad de los predictores."""
    etiquetas = {
        "prediccion_lstm": "Pronóstico base LSTM",
        "precio_lag_1": "Precio t", "precio_lag_2": "Precio t-1",
        "abastecimiento_lag_1": "Abastecimiento t",
        "precio_media_movil_7": "Precio: media 7 días",
        "precio_media_movil_30": "Precio: media 30 días",
        "precio_rango_movil_30": "Precio: rango 30 días",
        "dia_semana_sin": "Día semana (sen)", "dia_semana_cos": "Día semana (cos)",
        "mes_sin": "Mes (sen)", "mes_cos": "Mes (cos)",
        "dia_anio_sin": "Día año (sen)", "dia_anio_cos": "Día año (cos)",
    }
    if nombre in etiquetas: return etiquetas[nombre]
    return (nombre.replace("precio_lag_", "Precio rezago ")
            .replace("abastecimiento_lag_", "Abastecimiento rezago ")
            .replace("_", " "))


def figura_articulo_treeshap(model, X: np.ndarray, nombres: list[str],
                              vals: np.ndarray, esperado: float, posicion: int,
                              fecha: str, prefix: str, residuo: bool,
                              base_lstm: np.ndarray | None = None) -> None:
    """Beeswarm global y waterfall local a partir de SHAP REAL del modelo.

    Para el híbrido el waterfall explica sólo la corrección residual.
    """
    if vals.shape != X.shape or posicion < 0 or posicion >= len(X):
        raise ValueError("Forma o posición incorrecta en TreeSHAP del artículo")
    prediction = float(np.asarray(model.predict(X[posicion:posicion+1])).ravel()[0])
    reconstructed = float(esperado + vals[posicion].sum())
    if not np.isclose(prediction, reconstructed, atol=2e-3, rtol=2e-3):
        raise ValueError(f"TreeSHAP no reconstruye el modelo ({prefix}): "
                         f"{reconstructed} versus {prediction}")
    ranking = np.argsort(np.mean(np.abs(vals), axis=0))[::-1][:12]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 6.3),
                                  gridspec_kw={"width_ratios": [1.18, 1]})
    rng = np.random.default_rng(42)
    grafico_color = None
    for j, col in enumerate(ranking[::-1]):
        xs = X[:, col]
        p5, p95 = np.quantile(xs, [.05, .95])
        if p95 > p5:
            color = np.clip((xs-p5)/(p95-p5), 0, 1)
        else:
            color = np.full(len(xs), 0.5)
        grafico_color = ax1.scatter(vals[:, col], j+rng.uniform(-.24, .24, len(xs)),
                                   c=color, cmap="coolwarm", vmin=0, vmax=1,
                                   s=5, alpha=.55, rasterized=True)
    ax1.axvline(0, color="#4d4d4d", lw=.9)
    ax1.set_yticks(range(len(ranking)), [_rotulo_predictor(nombres[i]) for i in ranking[::-1]], fontsize=8)
    ax1.set(xlabel="Atribución TreeSHAP (S/kg)", title="(a) SHAP global - distribución de contribuciones")
    ax1.grid(axis="x", alpha=.14)
    fig.colorbar(grafico_color, ax=ax1, pad=.015, fraction=.04).set_label("Valor relativo del predictor: bajo a alto")
    local = vals[posicion]
    indices = np.argsort(np.abs(local))[::-1][:9]
    restantes = np.setdiff1d(np.arange(len(nombres)), indices)
    pasos = [(_rotulo_predictor(nombres[i]), float(local[i])) for i in indices]
    if len(restantes): pasos.append(("Otros predictores", float(local[restantes].sum())))
    acum = float(esperado)
    trazas = []
    for etiqueta, contrib in pasos:
        siguiente = acum + contrib
        trazas.append((etiqueta, contrib, acum, siguiente))
        acum = siguiente
    for j, (etiqueta, delta, desde, hasta) in enumerate(trazas[::-1]):
        ax2.barh(j, abs(delta), left=min(desde, hasta), height=.68,
                 color="#287e68" if delta >= 0 else "#c06150", alpha=.9)
        ax2.text(hasta, j, f" {delta:+.3f}", va="center", fontsize=7)
    ax2.set_yticks(range(len(trazas)), [r[0] for r in trazas[::-1]], fontsize=8)
    ax2.axvline(esperado, color="#808080", linestyle="--", lw=1,
                label=f"Base: {esperado:.3f}")
    ax2.axvline(prediction, color="#121212", lw=1.3,
                label=f"Predicción: {prediction:.3f}")
    ax2.legend(fontsize=8, loc="best")
    ax2.set(xlabel="Corrección residual (S/kg)" if residuo else "Precio pronosticado (S/kg)",
            title=f"(b) SHAP local - pronóstico del {fecha}")
    ax2.grid(axis="x", alpha=.14)
    title = "TreeSHAP de XGBoost" if not residuo else "TreeSHAP del corrector residual de LSTM-XGBoost"
    if "calendario_abastecimiento" in prefix:
        etiqueta_modelo = "Calendario + abastecimiento"
    elif "calendario" in prefix:
        etiqueta_modelo = "Calendario"
    elif "historico_abastecimiento" in prefix:
        etiqueta_modelo = "Histórico + abastecimiento"
    else:
        etiqueta_modelo = "Histórico"
    fig.suptitle(f"{title} - {etiqueta_modelo}", fontsize=13)
    if residuo and base_lstm is not None:
        base_pron = float(base_lstm[posicion])
        fig.text(.5, .012, f"Pronóstico LSTM: {base_pron:.3f} + corrección: {prediction:+.3f} "
                 f"= pronóstico híbrido: {base_pron+prediction:.3f} S/kg", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04 if residuo else 0, 1, .94))
    _guardar_figura(fig, f"Figura_articulo_TreeSHAP_global_local_{prefix}.png")


def figura_articulo_timeshap(prefix: Path, slug: str) -> None:
    """Global/local por variables e instantes; únicamente 30 secuencias explicadas."""
    path = lambda suffix: prefix.with_name(prefix.name + "_" + suffix + ".csv")
    cg = pd.read_csv(path("caracteristicas_global"))
    eg = pd.read_csv(path("eventos_global"))
    cl = pd.read_csv(path("caracteristicas_local"))
    el = pd.read_csv(path("eventos_local"))
    if any(x.empty for x in (cg, eg, cl, el)):
        raise ValueError("Faltan explicaciones TimeSHAP globales o locales")
    fechas = sorted(pd.to_datetime(cl.fecha_target.dropna().unique()))
    fecha = fechas[len(fechas)//2]
    loc_feat = cl.loc[pd.to_datetime(cl.fecha_target).eq(fecha)]
    loc_event = el.loc[pd.to_datetime(el.fecha_target).eq(fecha)]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9),
                            gridspec_kw={"width_ratios": [1, 1.12]})
    for ax, frame, field, titulo, signed in (
        (axs[0,0], cg, "atribucion_abs_media", "(a) Global - características", False),
        (axs[0,1], eg, "atribucion_abs_media", "(b) Global - instantes", False),
        (axs[1,0], loc_feat, "Shapley Value", "(c) Local - características", True),
        (axs[1,1], loc_event, "Shapley Value", "(d) Local - instantes", True),
    ):
        frame = frame.copy()
        if signed:
            frame = frame.groupby("Feature", as_index=False)[field].sum()
            frame = frame.reindex(frame[field].abs().sort_values(ascending=False).index)
        else:
            frame = frame.sort_values(field, ascending=False)
        frame = frame.head(12).iloc[::-1]
        nombres = frame.Feature.astype(str).map(lambda x: _rotulo_predictor(x).replace("Event -", "Instante -"))
        value = frame[field].to_numpy(dtype=float)
        if signed:
            ax.barh(nombres, value, color=["#287e68" if x>=0 else "#c06150" for x in value])
            ax.axvline(0, color="#333333", lw=.85)
        else:
            ax.barh(nombres, value, color="#3977a5")
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Atribución TimeSHAP (S/kg)" if signed else "Media |TimeSHAP| (S/kg)")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle(f"Explicabilidad global y local de LSTM - {'Calendario + abastecimiento' if slug == 'calendario_abastecimiento' else 'Calendario' if slug == 'calendario' else 'Histórico + abastecimiento' if slug == 'historico_abastecimiento' else 'Histórico'}", fontsize=13)
    fig.text(.5, .018, f"Global: muestra de {len(fechas)} secuencias. Local: ejemplo del {fecha:%d/%m/%Y}.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    _guardar_figura(fig, f"Figura_articulo_TimeSHAP_global_local_{slug}.png")


def figura_articulo_tcn_ig(local: pd.DataFrame, slug: str) -> None:
    """Mapas global/local calculados a partir de las mismas atribuciones IG."""
    fechas = sorted(pd.to_datetime(local.fecha_target.dropna().unique()))
    if not fechas:
        raise ValueError("No hay atribuciones locales IG disponibles")
    fecha = fechas[len(fechas)//2]
    datos = local.copy()
    datos["variable_corta"] = datos.variable.map(_rotulo_predictor)
    global_avg = datos.groupby(["variable_corta", "rezago_desde_origen"]).atribucion_abs.mean().unstack()
    local_sample = datos.loc[pd.to_datetime(datos.fecha_target).eq(fecha)]
    local_map = local_sample.groupby(["variable_corta", "rezago_desde_origen"]).atribucion.sum().unstack()
    orden = global_avg.sum(axis=1).sort_values(ascending=False).index
    global_avg = global_avg.reindex(index=orden).sort_index(axis=1, ascending=False)
    local_map = local_map.reindex(index=orden, columns=global_avg.columns).fillna(0)
    fig, axs = plt.subplots(1, 2, figsize=(15.5, max(5.0, .55*len(orden)+2)))
    max_global = max(float(global_avg.to_numpy().max()), 1e-9)
    lim_local = max(float(np.abs(local_map.to_numpy()).max()), 1e-9)
    im1 = axs[0].imshow(global_avg.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max_global)
    im2 = axs[1].imshow(local_map.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-lim_local, vmax=lim_local)
    for ax, data in zip(axs, (global_avg, local_map)):
        ax.set_yticks(range(len(orden)), list(orden), fontsize=8)
        n = len(data.columns)
        ticks = np.unique(np.linspace(0, n-1, min(n, 9), dtype=int))
        ax.set_xticks(ticks, [str(data.columns[t]) for t in ticks])
        ax.set_xlabel("Rezago respecto al origen t (días)")
    axs[0].set_title("(a) Global - magnitud media de atribuciones")
    axs[1].set_title(f"(b) Local - predicción del {fecha:%d/%m/%Y}")
    fig.colorbar(im1, ax=axs[0], fraction=.035, pad=.02, label="Media |atribución| (S/kg)")
    fig.colorbar(im2, ax=axs[1], fraction=.035, pad=.02, label="Atribución firmada (S/kg)")
    fig.suptitle(f"Integrated Gradients de TCN - {'Calendario + abastecimiento' if slug == 'calendario_abastecimiento' else 'Calendario' if slug == 'calendario' else 'Histórico + abastecimiento' if slug == 'historico_abastecimiento' else 'Histórico'}", fontsize=13)
    fig.text(.5, .012, f"Global: {len(fechas)} secuencias. Local: ejemplo del {fecha:%d/%m/%Y}.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    _guardar_figura(fig, f"Figura_articulo_IG_global_local_{slug}.png")


def generar_figuras_treeshap(model, X:np.ndarray, nombres:list[str], prefix:str,
                              fechas:np.ndarray, real:np.ndarray, pred:np.ndarray,
                              residuo:bool=False, base_lstm:np.ndarray | None=None) -> None:
    """Explicación global de todos los días y dos casos locales (mediano/extremo)."""
    import shap
    exp=shap.TreeExplainer(model)
    vals=np.asarray(exp.shap_values(X),dtype=float)
    if vals.ndim==3: vals=vals[...,0]
    if vals.shape!=(len(X),len(nombres)):
        raise ValueError(f"Atribuciones TreeSHAP inválidas para {prefix}")
    glo=pd.DataFrame({"predictor":nombres,"SHAP_medio":np.mean(vals,axis=0),
                      "SHAP_abs_medio":np.mean(np.abs(vals),axis=0)})
    glo.sort_values("SHAP_abs_medio",ascending=False).to_csv(
        TABLE_DIR/f"Tabla_SHAP_global_{prefix}.csv",index=False,encoding="utf-8-sig")
    top=np.argsort(np.mean(np.abs(vals),axis=0))[-15:][::-1]
    fig,ax=plt.subplots(figsize=(9,max(4, .35*len(top)+1.5)))
    ax.barh(np.arange(len(top)),np.mean(np.abs(vals[:,top]),axis=0)[::-1])
    ax.set_yticks(np.arange(len(top)),[nombres[i] for i in top[::-1]],fontsize=9)
    ax.set(xlabel="Media |TreeSHAP| (S/kg)",title=f"TreeSHAP global - {prefix.replace('_',' ')}")
    ax.grid(axis="x",alpha=.18);fig.tight_layout()
    _guardar_figura(fig,f"Figura_6_6_SHAP_global_{prefix}.png")
    fig,ax=plt.subplots(figsize=(9,max(4,.40*min(12,len(top))+1.8)))
    rng=np.random.default_rng(42);pos=top[:12]
    scatter=None
    for i,feature in enumerate(pos[::-1]):
        xvals=X[:,feature]
        lo,hi=np.quantile(xvals,[.05,.95]);den=max(float(hi-lo),1e-12)
        standardized=np.clip((xvals-lo)/den,0,1)
        jitter=rng.uniform(-.23,.23,len(vals))
        scatter=ax.scatter(vals[:,feature],np.full(len(vals),i)+jitter,c=standardized,
                           cmap="coolwarm",vmin=0,vmax=1,s=4,alpha=.5,rasterized=True)
    ax.axvline(0,color="gray",lw=.8)
    ax.set_yticks(range(len(pos)),[nombres[i] for i in pos[::-1]],fontsize=8)
    ax.set(xlabel="Contribución TreeSHAP (S/kg)",title=f"Dirección de efectos - {prefix.replace('_',' ')}")
    fig.colorbar(scatter,ax=ax,label="Valor de característica (normalizado por predictor)")
    fig.tight_layout();_guardar_figura(fig,f"Figura_6_6b_SHAP_beeswarm_{prefix}.png")
    valid=np.isfinite(real)
    idx=np.where(valid)[0]
    if not len(idx):return
    errors=np.abs(real[idx]-pred[idx]);med=idx[np.argsort(errors)[len(idx)//2]]
    extreme=idx[np.argmax(errors)]
    base=float(np.asarray(exp.expected_value).ravel()[0])
    summary=[]
    for case,position in (("representativo",int(med)),("error_alto",int(extreme))):
        contributions=vals[position];order=np.argsort(np.abs(contributions))[::-1]
        show=order[:min(10,len(order))]
        rest=float(contributions[order[len(show):]].sum())
        labels=[nombres[j] for j in show]+(["Otros predictores"] if len(order)>len(show) else [])
        parts=[float(contributions[j]) for j in show]+([rest] if len(order)>len(show) else [])
        cumulative=base;steps=[]
        for label,delta in zip(labels,parts):
            steps.append((label,delta,cumulative))
            cumulative+=delta
        fig,ax=plt.subplots(figsize=(10,max(4.4,.38*len(steps)+2)))
        for i,(label,delta,before) in enumerate(steps[::-1]):
            ax.barh(i,abs(delta),left=min(before,before+delta),height=.65,
                    color="#267f69" if delta>0 else "#be624c")
            ax.text(before+delta,i,f" {delta:+.3f}",fontsize=8,va="center")
        ax.set_yticks(range(len(steps)),[item[0] for item in steps[::-1]],fontsize=8)
        ax.axvline(base,color="gray",ls="--",lw=1,label=f"Base={base:.3f}")
        pred_model=float(np.asarray(model.predict(X[position:position+1])).ravel()[0])
        ax.axvline(pred_model,color="black",lw=1.2,label=f"Modelo={pred_model:.3f}")
        fecha=pd.Timestamp(fechas[position]).strftime('%Y-%m-%d')
        target="residuo de LSTM" if residuo else "precio"
        ax.set(xlabel=f"Predicción de {target} (S/kg)",
               title=f"TreeSHAP local: {case} - {prefix}, objetivo {fecha}\n"
                     f"Real={real[position]:.3f}; pronóstico={pred[position]:.3f} S/kg")
        ax.legend(fontsize=8);ax.grid(axis="x",alpha=.16);fig.tight_layout()
        _guardar_figura(fig,f"Figura_6_7_SHAP_local_{prefix}_{case}.png")
        summary.append({"caso":case,"fecha_objetivo":fecha,"real":real[position],
                        "pronostico":pred[position],"valor_esperado":base,
                        "prediccion_desde_atribuciones":cumulative,
                        "prediccion_del_modelo":pred_model,"error_reconstruccion":cumulative-pred_model})
    pd.DataFrame(summary).to_csv(TABLE_DIR/f"Tabla_SHAP_local_{prefix}.csv",index=False,encoding="utf-8-sig")
    figura_articulo_treeshap(model, X, nombres, vals, base, int(med),
                            pd.Timestamp(fechas[int(med)]).strftime("%d/%m/%Y"),
                            prefix, residuo, base_lstm=base_lstm)


def generar_figuras_timeshap(prefix:Path,slug:str) -> None:
    """Global por características y eventos; local de un día realmente explicado."""
    for suffix,category in (("caracteristicas_global","características"),("eventos_global","instantes de la ventana")):
        data=pd.read_csv(prefix.with_name(prefix.name+f"_{suffix}.csv"))
        _figura_atribuciones_globales(data,f"TimeSHAP global - LSTM {slug} ({category})",
            f"Figura_6_8_TimeSHAP_global_{slug}_{suffix}.png", "Feature")
    for suffix,category in (("caracteristicas_local","caracteristicas"),("eventos_local","eventos")):
        local=pd.read_csv(prefix.with_name(prefix.name+f"_{suffix}.csv"))
        if local.empty:continue
        fechas=sorted(pd.to_datetime(local["fecha_target"].dropna().unique()))
        elegido=fechas[len(fechas)//2]
        subset=local[pd.to_datetime(local["fecha_target"]).eq(elegido)]
        _figura_local_barras(subset,f"TimeSHAP local - LSTM {slug}, {elegido:%Y-%m-%d}",
                  f"Figura_6_8_TimeSHAP_local_{slug}_{category}.png")
    figura_articulo_timeshap(prefix, slug)


def generar_figuras_tcn_ig(prefix:Path,slug:str) -> None:
    variables=pd.read_csv(prefix.with_name(prefix.name+"_variables.csv"))
    temporal=pd.read_csv(prefix.with_name(prefix.name+"_temporal.csv"))
    local=pd.read_csv(prefix.with_name(prefix.name+"_local.csv"))
    _figura_atribuciones_globales(variables,f"Integrated Gradients: TCN {slug} - variables",
        f"Figura_6_9_IG_global_variables_{slug}.png","variable")
    fig,ax=plt.subplots(figsize=(8.8,4.4))
    ax.plot(temporal["rezago_desde_origen"],temporal["atribucion_abs_media"],marker="o",ms=3)
    ax.set(xlabel="Rezago respecto al origen t (días)",ylabel="Media |atribución| (S/kg)",
           title=f"Integrated Gradients global - TCN {slug}: importancia temporal")
    ax.invert_xaxis();ax.grid(alpha=.16);fig.tight_layout()
    _guardar_figura(fig,f"Figura_6_9b_IG_global_rezagos_{slug}.png")
    if not local.empty:
        fechas=sorted(pd.to_datetime(local["fecha_target"].dropna().unique()))
        fecha=fechas[len(fechas)//2]
        z=local.loc[pd.to_datetime(local["fecha_target"]).eq(fecha)]
        pivot=z.pivot_table(index="variable",columns="rezago_desde_origen",values="atribucion",aggfunc="sum").sort_index(axis=1,ascending=False)
        fig,ax=plt.subplots(figsize=(11,max(4.5,len(pivot)*.5+2)))
        vmax=max(np.max(np.abs(pivot.to_numpy(dtype=float))),1e-7)
        img=ax.imshow(pivot.to_numpy(dtype=float),aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        ax.set_yticks(range(len(pivot.index)),pivot.index,fontsize=8)
        positions=np.linspace(0,len(pivot.columns)-1,min(10,len(pivot.columns))).astype(int)
        ax.set_xticks(positions,[str(pivot.columns[i]) for i in positions])
        ax.set(xlabel="Rezago desde origen t (días)",title=f"TCN {slug}: atribución local de un pronóstico ({fecha:%Y-%m-%d})")
        fig.colorbar(img,ax=ax,label="Atribución firmada (S/kg)")
        fig.tight_layout();_guardar_figura(fig,f"Figura_6_9c_IG_local_mapa_{slug}.png")
    figura_articulo_tcn_ig(local, slug)


def generar_figura_sarimax_parametros(table:pd.DataFrame,slug:str) -> None:
    data=table.loc[~table["parametro"].str.contains("sigma2",case=False,na=False)]
    if data.empty:return
    fig,ax=plt.subplots(figsize=(8,max(3,len(data)*.45+1.5)))
    ax.barh(data["parametro"],data["coeficiente"])
    ax.axvline(0,color="black",lw=.8)
    ax.set(xlabel="Coeficiente estimado (escala del parámetro)",title=f"SARIMAX: parámetros estimados - {slug}")
    ax.grid(axis="x",alpha=.17);fig.tight_layout()
    _guardar_figura(fig,f"Figura_6_10_SARIMAX_parametros_{slug}.png")


def pruebas_automaticas_metricas() -> None:
    """Pruebas sintéticas deterministas; no consultan prueba final ni cambian modelos."""
    truth=np.array([1.,2.,3.]);perfect=truth.copy()
    assert abs(smape(truth,perfect))<1e-12
    perfect_metrics=calcular_metricas_test(truth,perfect,1.)
    assert all(abs(perfect_metrics[k])<1e-12 for k in ("smape","mae","rmse","mase"))
    assert abs(perfect_metrics["r2"]-1.)<1e-12
    assert abs(smape(np.array([2.]),np.array([1.]))-200/3)<1e-10
    same=contraste_diebold_mariano_hac(truth,perfect,perfect+1e-12)
    assert same["equivalencia_numerica"] and same["p_valor"]==1
    table=pd.DataFrame({"real":[1.,np.nan,2.,3.]})
    for model in ("Naive","SeasonalNaive_7"):
        table[model]=[1.,1.,2.,3.]
    for cfg in CONFIGURATIONS:
        for model in MODELS:table[clave_modelo(model,cfg)]=[1.,1.,2.,3.]
    metrics=construir_tabla_metricas(table,1.)
    assert metrics["n"].eq(3).all() and metrics["objetivos_ausentes"].eq(1).all()
    dm=contraste_diebold_mariano_hac(table.real.to_numpy(),table.Naive.to_numpy(),table.Naive.to_numpy())
    assert dm["n"]==3 and dm["pares_temporales_ausentes"]==1
    print("Pruebas de métricas, NaN y equivalencia numérica: CORRECTAS.")


def guardar_sensibilidad_banda_DM(prediction_table:pd.DataFrame) -> None:
    """Diagnóstico HAC alternativo, NO cambia el resultado confirmatorio."""
    y=prediction_table["real"].to_numpy(dtype=float)
    n=int(np.isfinite(y).sum())
    lags=sorted(set([0,calcular_nlags_hac(n),min(7,n-1),min(14,n-1)]))
    rows=[]
    for cfg in CONFIGURATIONS:
        for model in MODELS:
            a=prediction_table[clave_modelo(model,cfg)].to_numpy(dtype=float)
            b=prediction_table["Naive"].to_numpy(dtype=float)
            valid=np.isfinite(y)
            diff=(y[valid]-a[valid])**2-(y[valid]-b[valid])**2
            if np.allclose(a[valid],b[valid],rtol=1e-9,atol=1e-9):
                for lag in lags:
                    rows.append({"modelo":model,"configuracion":etiqueta_configuracion(cfg),
                        "lag_HAC":lag,"diferencial_MSE":0.,"p_sin_ajustar":1.,"equivalente":True})
                continue
            for lag in lags:
                result=sm.OLS(diff,np.ones((len(diff),1))).fit()
                v=float(cov_hac(result,nlags=lag,use_correction=True)[0,0])
                statistic=float(np.mean(diff)/np.sqrt(v)) if v>0 else float("nan")
                rows.append({"modelo":model,"configuracion":etiqueta_configuracion(cfg),
                    "lag_HAC":lag,"diferencial_MSE":float(np.mean(diff)),
                    "p_sin_ajustar":float(2*norm.sf(abs(statistic))) if np.isfinite(statistic) else float("nan"),
                    "equivalente":False})
    pd.DataFrame(rows).to_csv(TABLE_DIR/"Tabla_6c_sensibilidad_HAC_descriptiva.csv",index=False,encoding="utf-8-sig")
    print("Sensibilidad del ancho de banda HAC: descriptiva, sin reajustar hipótesis.")



def main() -> None:
    inicio = perf_counter()
    fijar_semilla(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pruebas_automaticas_metricas()
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    guardar_json(STATE_FILE, {"estado":"EN_CURSO","inicio":datetime.now().isoformat(timespec="seconds")})
    verificar_dependencias_explicabilidad()

    imprimir_titulo("FASE 4 - EVALUACIÓN FINAL, COMPARACIÓN ESTADÍSTICA E INTERPRETACIÓN")
    print(f"Dispositivo PyTorch: {DEVICE}", flush=True)
    print(f"Dataset: {INPUT_FILE}", flush=True)
    print(f"Artefactos de validación: {VALIDATION_DIR}", flush=True)

    df = cargar_dataset()
    partition_metadata = cargar_json(PARTITION_FILE)
    best_configs = cargar_json(BEST_CONFIG_FILE)
    artifacts = cargar_json(ARTIFACT_FILE)
    implementation_idx, test_idx = reconstruir_particion(df, partition_metadata)
    y_test = df.loc[test_idx, TARGET].to_numpy(dtype=float)
    print(f"Precios objetivo observados en prueba: {int(np.isfinite(y_test).sum())}/{len(y_test)}",flush=True)

    print(
        f"Conjunto final de prueba: {len(test_idx)} observaciones | "
        f"origen {df.loc[test_idx[0], DATE].date()} a {df.loc[test_idx[-1], DATE].date()} | "
        f"objetivo {df.loc[test_idx[0], TARGET_DATE].date()} a {df.loc[test_idx[-1], TARGET_DATE].date()}",
        flush=True,
    )

    prediction_table = pd.DataFrame({
        "fecha_origen": df.loc[test_idx, DATE].to_numpy(),
        "fecha_target": df.loc[test_idx, TARGET_DATE].to_numpy(),
        "real": y_test,
        "objetivo_observado": np.isfinite(y_test).astype(int),
    })
    predictions: dict[str, np.ndarray] = {}
    explain_resources: dict[str, dict] = {}

    imprimir_subtitulo("LÍNEAS BASE")
    baselines = predecir_baselines(df, test_idx)
    predictions.update(baselines)
    for name, values in baselines.items():
        prediction_table[name] = values
        print(f"{name}: {len(values)} pronósticos.", flush=True)

    for configuracion in CONFIGURATIONS:
        etiqueta = etiqueta_configuracion(configuracion)
        slug = slug_configuracion(configuracion)
        imprimir_titulo(f"EVALUACIÓN FINAL - {etiqueta}")

        sarimax_result, sarimax_scaler, sarimax_info = cargar_sarimax(artifacts, configuracion)
        sarimax_pred = predecir_sarimax_test(sarimax_result, sarimax_scaler, df, test_idx, configuracion)
        predictions[clave_modelo("SARIMAX", configuracion)] = sarimax_pred
        prediction_table[clave_modelo("SARIMAX", configuracion)] = sarimax_pred
        explain_resources[clave_modelo("SARIMAX", configuracion)] = {"result": sarimax_result, "info": sarimax_info}
        print(f"SARIMAX - {etiqueta}: {len(sarimax_pred)} pronósticos.", flush=True)

        xgb_model, xgb_info = cargar_xgboost(artifacts, configuracion)
        xgb_pred, X_xgb, xgb_columns = predecir_xgboost_test(xgb_model, df, test_idx, configuracion)
        predictions[clave_modelo("XGBoost", configuracion)] = xgb_pred
        prediction_table[clave_modelo("XGBoost", configuracion)] = xgb_pred
        explain_resources[clave_modelo("XGBoost", configuracion)] = {"model": xgb_model, "X_test": X_xgb, "columns": xgb_columns, "info": xgb_info}
        print(f"XGBoost - {etiqueta}: {len(xgb_pred)} pronósticos.", flush=True)

        lstm_model, lstm_params, lstm_scalers = cargar_lstm(artifacts, configuracion)
        lstm_pred, lstm_test_data = predecir_lstm_test(lstm_model, lstm_params, lstm_scalers, df, test_idx, configuracion)
        predictions[clave_modelo("LSTM", configuracion)] = lstm_pred
        prediction_table[clave_modelo("LSTM", configuracion)] = lstm_pred
        X_lstm_train, y_lstm_train, _ = construir_ventanas(df, implementation_idx, int(lstm_params["window"]), configuracion)
        X_lstm_train_scaled, _ = escalar_secuencias(X_lstm_train, y_lstm_train, lstm_scalers["x_scaler"], lstm_scalers["y_scaler"])
        explain_resources[clave_modelo("LSTM", configuracion)] = {
            "model": lstm_model, "params": lstm_params, "scalers": lstm_scalers,
            "test": lstm_test_data, "X_train_scaled": X_lstm_train_scaled,
        }
        print(f"LSTM - {etiqueta}: {len(lstm_pred)} pronósticos.", flush=True)

        tcn_model, tcn_params, tcn_scalers = cargar_tcn(artifacts, configuracion)
        tcn_pred, tcn_test_data = predecir_tcn_test(tcn_model, tcn_params, tcn_scalers, df, test_idx, configuracion)
        predictions[clave_modelo("TCN", configuracion)] = tcn_pred
        prediction_table[clave_modelo("TCN", configuracion)] = tcn_pred
        X_tcn_train, y_tcn_train, _ = construir_ventanas(df, implementation_idx, int(tcn_params["window"]), configuracion, usar_calendario=bool(tcn_params.get("usar_calendario",True)))
        X_tcn_train_scaled, _ = escalar_secuencias(X_tcn_train, y_tcn_train, tcn_scalers["x_scaler"], tcn_scalers["y_scaler"])
        explain_resources[clave_modelo("TCN", configuracion)] = {
            "model": tcn_model, "params": tcn_params, "scalers": tcn_scalers,
            "test": tcn_test_data, "X_train_scaled": X_tcn_train_scaled,
        }
        print(f"TCN - {etiqueta}: {len(tcn_pred)} pronósticos.", flush=True)

        residual_model, hybrid_info = cargar_corrector_hibrido(artifacts, configuracion)
        hybrid_pred, X_res_test, residual_columns = predecir_hibrido_test(residual_model, df, test_idx, configuracion, lstm_pred)
        predictions[clave_modelo("LSTM-XGBoost", configuracion)] = hybrid_pred
        prediction_table[clave_modelo("LSTM-XGBoost", configuracion)] = hybrid_pred
        explain_resources[clave_modelo("LSTM-XGBoost", configuracion)] = {
            "residual_model": residual_model, "X_res_test": X_res_test,
            "columns": residual_columns, "info": hybrid_info,
        }
        print(f"LSTM-XGBoost - {etiqueta}: {len(hybrid_pred)} pronósticos.", flush=True)

    validar_predicciones(predictions, len(test_idx))
    expected_columns = [clave_modelo(model, configuracion) for configuracion in CONFIGURATIONS for model in MODELS]
    if prediction_table[expected_columns].isna().any().any():
        raise ValueError("Existen pronósticos faltantes en la tabla emparejada de la evaluación final.")

    prediction_table.to_csv(OUTPUT_DIR / "predicciones_evaluacion_final.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    imprimir_titulo("MÉTRICAS DE EVALUACIÓN FINAL")
    mase_scale = calcular_escala_mase(df, implementation_idx)
    metrics = construir_tabla_metricas(prediction_table, mase_scale)
    metrics.to_csv(OUTPUT_DIR / "metricas_evaluacion_final.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False), flush=True)

    comparacion = construir_comparacion_historico_abastecimiento(metrics)
    comparacion.to_csv(OUTPUT_DIR / "comparacion_historico_vs_historico_abastecimiento.csv", index=False, encoding="utf-8-sig")

    imprimir_titulo("DIEBOLD-MARIANO CON VARIANZA HAC Y CORRECCIÓN DE HOLM")
    dm_historico = comparar_modelos_dm(prediction_table, CONFIG_HISTORICO)
    dm_abastecimiento = comparar_modelos_dm(prediction_table, CONFIG_HISTORICO_ABASTECIMIENTO)
    dm_comparacion = comparar_historico_abastecimiento_dm(prediction_table)
    dm_baselines = pd.concat([comparar_baselines_dm(prediction_table,cfg)
                              for cfg in CONFIGURATIONS],ignore_index=True)
    dm_historico.to_csv(OUTPUT_DIR / "dm_modelos_historico.csv", index=False, encoding="utf-8-sig")
    dm_abastecimiento.to_csv(OUTPUT_DIR / "dm_modelos_historico_abastecimiento.csv", index=False, encoding="utf-8-sig")
    dm_comparacion.to_csv(OUTPUT_DIR / "dm_historico_vs_historico_abastecimiento.csv", index=False, encoding="utf-8-sig")
    dm_baselines.to_csv(OUTPUT_DIR / "dm_modelos_vs_baselines.csv", index=False, encoding="utf-8-sig")
    print("Familia modelos - Histórico:", flush=True)
    print(dm_historico.to_string(index=False), flush=True)
    print("\nFamilia modelos - Histórico + abastecimiento:", flush=True)
    print(dm_abastecimiento.to_string(index=False), flush=True)
    print("\nHistórico vs. Histórico + abastecimiento:", flush=True)
    print(dm_comparacion.to_string(index=False), flush=True)
    guardar_sensibilidad_banda_DM(prediction_table)
    generar_tablas_y_figuras_resultados(prediction_table, metrics, comparacion,
                                        dm_historico, dm_abastecimiento, dm_comparacion, dm_baselines)

    imprimir_titulo("INTERPRETACIÓN DE MODELOS")
    explain_status: dict[str, str] = {}
    target_dates = prediction_table["fecha_target"].to_numpy()

    for configuracion in CONFIGURATIONS:
        etiqueta = etiqueta_configuracion(configuracion)
        slug = slug_configuracion(configuracion)
        imprimir_subtitulo(f"TreeSHAP - XGBoost - {etiqueta}")
        xgb_resource = explain_resources[clave_modelo("XGBoost", configuracion)]
        explicar_xgboost_treeshap(xgb_resource["model"], xgb_resource["X_test"], xgb_resource["columns"], EXPLAIN_DIR / f"xgboost_{slug}_treeshap.csv")
        generar_figuras_treeshap(
            xgb_resource["model"], xgb_resource["X_test"], xgb_resource["columns"],
            f"xgboost_{slug}", target_dates, y_test,
            prediction_table[clave_modelo("XGBoost", configuracion)].to_numpy(dtype=float))
        explain_status[clave_modelo("XGBoost", configuracion)] = "TreeSHAP completado"

        imprimir_subtitulo(f"TimeSHAP - LSTM - {etiqueta}")
        lstm_resource = explain_resources[clave_modelo("LSTM", configuracion)]
        if OMITIR_TIMESHAP:
            explain_status[clave_modelo("LSTM", configuracion)] = "TimeSHAP omitido por FASE4_OMITIR_TIMESHAP=1"
        else:
            feature_names = obtener_nombres_caracteristicas_secuenciales(configuracion)
            explicar_lstm_timeshap(
                lstm_resource["model"], lstm_resource["X_train_scaled"], lstm_resource["test"]["X_scaled"],
                feature_names, lstm_resource["scalers"]["y_scaler"], target_dates,
                EXPLAIN_DIR / f"lstm_{slug}_timeshap",
            )
            registrar_componente_lstm_hibrido(configuracion)
            generar_figuras_timeshap(EXPLAIN_DIR / f"lstm_{slug}_timeshap", slug)
            explain_status[clave_modelo("LSTM", configuracion)] = "TimeSHAP completado"
            explain_status[f"{clave_modelo('LSTM-XGBoost', configuracion)}_LSTM"] = "Reutiliza TimeSHAP de la misma LSTM base"

        imprimir_subtitulo(f"Integrated Gradients - TCN - {etiqueta}")
        tcn_resource = explain_resources[clave_modelo("TCN", configuracion)]
        feature_names = obtener_nombres_caracteristicas_secuenciales(configuracion, usar_calendario=bool(tcn_resource["params"].get("usar_calendario",True)))
        explicar_tcn_integrated_gradients(
            tcn_resource["model"], tcn_resource["X_train_scaled"], tcn_resource["test"]["X_scaled"],
            feature_names, tcn_resource["scalers"]["y_scaler"], target_dates,
            EXPLAIN_DIR / f"tcn_{slug}_integrated_gradients",
        )
        generar_figuras_tcn_ig(EXPLAIN_DIR / f"tcn_{slug}_integrated_gradients", slug)
        explain_status[clave_modelo("TCN", configuracion)] = "Integrated Gradients completado"

        imprimir_subtitulo(f"Coeficientes - SARIMAX - {etiqueta}")
        sarimax_resource = explain_resources[clave_modelo("SARIMAX", configuracion)]
        tabla_sarimax = explicar_sarimax(sarimax_resource["result"], configuracion, EXPLAIN_DIR / f"sarimax_{slug}_parametros.csv")
        generar_figura_sarimax_parametros(tabla_sarimax, slug)
        explain_status[clave_modelo("SARIMAX", configuracion)] = "Parámetros exportados"

        imprimir_subtitulo(f"TreeSHAP - corrector residual LSTM-XGBoost - {etiqueta}")
        hybrid_resource = explain_resources[clave_modelo("LSTM-XGBoost", configuracion)]
        explicar_xgboost_treeshap(
            hybrid_resource["residual_model"], hybrid_resource["X_res_test"], hybrid_resource["columns"],
            EXPLAIN_DIR / f"hibrido_xgboost_{slug}_treeshap.csv",
        )
        base_lstm = prediction_table[clave_modelo("LSTM",configuracion)].to_numpy(dtype=float)
        residuo_real = y_test - base_lstm
        correccion_pred = prediction_table[clave_modelo("LSTM-XGBoost",configuracion)].to_numpy(dtype=float) - base_lstm
        generar_figuras_treeshap(
            hybrid_resource["residual_model"], hybrid_resource["X_res_test"], hybrid_resource["columns"],
            f"hibrido_corrector_{slug}", target_dates, residuo_real, correccion_pred,
            residuo=True, base_lstm=base_lstm)
        explain_status[f"{clave_modelo('LSTM-XGBoost', configuracion)}_XGBoost"] = "TreeSHAP del corrector residual completado"

    # La ablación reutiliza los checkpoints seleccionados en los cinco folds.
    # Añade tablas/figuras y nuevas columnas de pronóstico, sin alterar
    # los contrastes y figuras originales de las dos configuraciones principales.
    from fase4b_ablacion import ejecutar_ablacion as ejecutar_ablacion_fase4
    ejecutar_ablacion_fase4(prediction_table, df, implementation_idx, test_idx,
                           artifacts, mase_scale)

    metadata = {
        "fase": 4,
        "nombre": "Evaluación final, comparación estadística e interpretación",
        "dataset": str(INPUT_FILE),
        "n_implementacion": int(len(implementation_idx)),
        "n_prueba": int(len(test_idx)),
        "n_objetivos_observados_prueba":int(np.isfinite(y_test).sum()),
        "n_objetivos_faltantes_prueba":int((~np.isfinite(y_test)).sum()),
        "sha256_dataset_fase2":hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest(),
        "origen_prueba_inicio": str(df.loc[test_idx[0], DATE].date()),
        "origen_prueba_fin": str(df.loc[test_idx[-1], DATE].date()),
        "objetivo_prueba_inicio": str(df.loc[test_idx[0], TARGET_DATE].date()),
        "objetivo_prueba_fin": str(df.loc[test_idx[-1], TARGET_DATE].date()),
        "horizonte": "un día calendario",
        "metricas": {
            "principal": "sMAPE",
            "complementarias": ["MAE", "RMSE", "MASE", "R2"],
            "mase_referencia": "Naïve no estacional de un paso sobre el conjunto de implementación",
            "mase_escala": mase_scale,
        },
        "diebold_mariano": {
            "funcion_perdida": "error cuadrático",
            "varianza": "HAC Newey-West con kernel Bartlett",
            "regla_nlags": "floor(4*(n/100)^(2/9)), acotado a n-1",
            "distribucion_referencia": "normal asintótica bilateral",
            "alpha": ALPHA,
            "familias_holm": {
                "modelos_historico": 10,
                "modelos_historico_abastecimiento": 10,
                "historico_vs_historico_abastecimiento": 5,
                "modelos_vs_dos_baselines_por_configuracion": 10,
            },
        },
        "ablacion": {"post_hoc": True, "escenarios": ["Calendario", "Calendario + abastecimiento"],
                     "modelos": ["XGBoost", "LSTM", "TCN", "LSTM-XGBoost"]},
        "explicabilidad": {
            "XGBoost": "TreeSHAP", "LSTM": "TimeSHAP", "TCN": "Integrated Gradients",
            "SARIMAX": "parámetros estimados",
            "LSTM-XGBoost": "TimeSHAP en LSTM base y TreeSHAP en corrector residual",
            "timeshap_nsamples": TIMESHAP_NSAMPLES, "timeshap_instancias": TIMESHAP_MAX_INSTANCES,
            "integrated_gradients_steps": IG_STEPS, "integrated_gradients_instancias": IG_MAX_INSTANCES,
            "timeshap_omitido": OMITIR_TIMESHAP, "estado": explain_status,
        },
        "mejores_configuraciones_fase3": best_configs,
        "versiones": obtener_versiones_librerias(),
    }
    metadata["figuras_generadas"] = [p.name for p in sorted(FIGURE_DIR.glob("*.png"))]
    metadata["tablas_generadas"] = [p.name for p in sorted(TABLE_DIR.glob("*.csv"))]
    pd.DataFrame({"figura":[p.name for p in sorted(FIGURE_DIR.glob("*.png"))]}).to_csv(
        OUTPUT_DIR / "catalogo_figuras_resultados.csv",index=False,encoding="utf-8-sig")
    guardar_json(OUTPUT_DIR / "metadata_evaluacion_final.json", metadata)

    elapsed = perf_counter() - inicio
    imprimir_titulo("FASE 4 COMPLETADA")
    print(f"Predicciones: {OUTPUT_DIR / 'predicciones_evaluacion_final.csv'}", flush=True)
    print(f"Métricas: {OUTPUT_DIR / 'metricas_evaluacion_final.csv'}", flush=True)
    print(f"Contrastes DM: {OUTPUT_DIR}", flush=True)
    print(f"Explicabilidad: {EXPLAIN_DIR}", flush=True)
    guardar_json(STATE_FILE, {"estado":"COMPLETADA" if not OMITIR_TIMESHAP else "COMPLETADA_SIN_TIMESHAP",
                             "fin":datetime.now().isoformat(timespec="seconds"),
                             "imagenes":str(FIGURE_DIR),"tablas":str(TABLE_DIR)})
    print(f"Tiempo total: {elapsed / 60.0:.1f} minutos.", flush=True)


if __name__ == "__main__":
    ejecutar_con_log(main, "FASE 4", LOG_FILE)
