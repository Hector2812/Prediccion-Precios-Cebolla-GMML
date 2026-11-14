from __future__ import annotations

import copy
import sys
import traceback
import json
import hashlib
import math
import random
import warnings
import os
import platform
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.dates as mdates
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data procesada" / "dataset_modelado_base.csv"
OUTPUT_DIR = ROOT / "outputs" / "validacion"
MODEL_DIR = OUTPUT_DIR / "modelos"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "Fase_3_log.txt"
FIGURE_DIR = ROOT / "Figuras" / "experimentacion"
HISTORY_DIR = OUTPUT_DIR / "historiales_experimentacion"
TABLE_DIR = ROOT / "outputs" / "tablas_experimentacion"

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
MISSING_DAYS_PRICE = "dias_sin_precio_observado"
MISSING_DAYS_VOLUME = "dias_sin_abastecimiento_observado"
PREPROCESSING_VERSION = "Fase2_preprocesamiento_causal_v4_seleccion_alpha"
PROTOCOL_FILE = ROOT / "data procesada" / "protocolo_preprocesamiento.json"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnosticos_sensibilidad"
EJECUTAR_DIAGNOSTICOS_AMPLIADOS = os.getenv("FASE3_DIAGNOSTICOS_AMPLIADOS", "1") == "1"


CALENDAR_COLUMNS = (
    "dia_semana_sin",
    "dia_semana_cos",
    "mes_sin",
    "mes_cos",
    "dia_anio_sin",
    "dia_anio_cos",
)


CONFIG_HISTORICO = "historico"
CONFIG_HISTORICO_ABASTECIMIENTO = "historico_abastecimiento"
CONFIGURACIONES = (CONFIG_HISTORICO, CONFIG_HISTORICO_ABASTECIMIENTO)
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

IMPLEMENTATION_RATIO = 0.80
N_SPLITS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMAPE_EPS = 1e-12

XGB_TRIALS = 30
LSTM_TRIALS = 20
TCN_TRIALS = 20
HYBRID_XGB_TRIALS = 30
HYBRID_NESTED_LSTM_SELECTION = True


LSTM_WINDOWS = (7, 14, 30, 60)
LSTM_BATCH_SIZES = (8, 32, 64)
LSTM_MAX_EPOCHS = 200
LSTM_PATIENCE = 15
LSTM_LEARNING_RATE = 1e-3

TCN_WINDOWS = (15, 30, 45, 60)
TCN_BLOCK_OPTIONS = (2, 3, 4, 5)
TCN_CHANNEL_OPTIONS = (32, 64, 128)
TCN_KERNEL_OPTIONS = (3, 5)
TCN_DROPOUT_OPTIONS = (0.2, 0.3, 0.4)
TCN_BATCH_SIZE = 64
TCN_MAX_EPOCHS = 100
TCN_PATIENCE = 10
TCN_LEARNING_RATE = 1e-3
TCN_AT_LIMIT_MARGIN = 1
TCN_ARCHITECTURE_ID = "TCN_residual_2conv_causal_v2"

SARIMAX_TOP_K = 12
SARIMAX_MAX_TOTAL_ARMA_ORDER = 2
SARIMAX_SEASONAL_PERIOD = 7
SARIMAX_MAXITER_LBFGS = 300
SARIMAX_MAXITER_POWELL = 500


@dataclass
class SarimaxCandidate:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float | None = None
    convergio: bool | None = None
    optimizador: str | None = None


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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        y = self.dropout1(self.relu1(self.conv1(x)))
        y = self.dropout2(self.relu2(self.conv2(y)))
        return self.final_relu(y + residual)


class TCNRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_channels: int,
        n_blocks: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = n_features
        for level in range(n_blocks):
            dilation = 2 ** level
            blocks.append(TemporalBlock(
                in_channels=in_channels, out_channels=hidden_channels,
                kernel_size=kernel_size, dilation=dilation, dropout=dropout,
            ))
            in_channels = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.output = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.network(y)
        return self.output(y[:, :, -1]).squeeze(-1)


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
        except KeyboardInterrupt:
            estado = "INTERRUMPIDA_POR_USUARIO"
            print("\nEjecución interrumpida manualmente con Ctrl+C.", flush=True)
            raise
        except Exception:
            estado = "ERROR"
            print("\nERROR NO CONTROLADO:", flush=True)
            traceback.print_exc()
            raise
        finally:
            fin = datetime.now()
            print(
                f"[{fin:%Y-%m-%d %H:%M:%S}] FIN {nombre_fase} | "
                f"estado={estado} | duración={fin - inicio}",
                flush=True,
            )
            sys.stdout = stdout_original
            sys.stderr = stderr_original


@contextmanager
def registrar_bloque(nombre: str):
    inicio_dt = datetime.now()
    inicio_perf = perf_counter()
    print(f"[{inicio_dt:%Y-%m-%d %H:%M:%S}] INICIO {nombre}", flush=True)
    try:
        yield
    finally:
        fin_dt = datetime.now()
        duracion = perf_counter() - inicio_perf
        print(
            f"[{fin_dt:%Y-%m-%d %H:%M:%S}] FIN {nombre} | "
            f"duración={duracion / 60.0:.2f} min",
            flush=True,
        )


def imprimir_titulo(texto: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(texto, flush=True)
    print("=" * 72, flush=True)


def imprimir_subtitulo(texto: str) -> None:
    print("\n" + "-" * 72, flush=True)
    print(texto, flush=True)
    print("-" * 72, flush=True)


def formatear_parametros(params: dict) -> str:
    return ", ".join(f"{clave}={valor}" for clave, valor in params.items())


def fijar_semilla(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not len(y_true) or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Las métricas requieren pares observados, finitos y no vacíos.")
    return {
        "smape": smape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def cargar_dataset() -> pd.DataFrame:
    """Carga estricta de Fase 2 causal; no modifica etiquetas originales.

    Las fechas y los predictores deben ser válidos. Solamente el target
    supervisado puede tener NaN, y exclusivamente donde SISAP no reportó precio.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No se encontró la salida de la Fase 2: {INPUT_FILE}")
    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError("Ejecute antes Fase 2: falta protocolo_preprocesamiento.json")
    protocolo = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocolo.get("version") != PREPROCESSING_VERSION:
        raise ValueError("Versión de preprocesamiento no coincide entre Fase 2 y Fase 3")
    if protocolo.get("regla_seleccion_alpha") != "minimo_MAE_promedio_equilibrado_por_duracion":
        raise ValueError("La Fase 2 no registró la selección de alpha esperada.")
    seleccion = PROTOCOL_FILE.parent / protocolo.get("seleccion_alpha_archivo", "")
    if not seleccion.is_file() or hashlib.sha256(seleccion.read_bytes()).hexdigest() != protocolo.get("sha256_seleccion_alpha"):
        raise ValueError("Falta o cambió el archivo auditado de selección EWMA.")
    evidencia = json.loads(seleccion.read_text(encoding="utf-8"))
    if not math.isclose(float(evidencia["alpha_ewma"]), float(protocolo["alpha_ewma"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("El alpha EWMA no coincide con su archivo de selección.")
    digest=hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest()
    if protocolo.get("sha256_dataset") != digest:
        raise ValueError("El dataset procesado no coincide con el protocolo de la Fase 2")
    df = pd.read_csv(INPUT_FILE)
    required = [
        DATE, TARGET_DATE, PRICE, VOLUME, TARGET, PRICE_ORIGINAL,
        VOLUME_ORIGINAL, TARGET_OBSERVED, PRICE_IMPUTED, VOLUME_IMPUTED,
        MISSING_DAYS_PRICE, MISSING_DAYS_VOLUME, *CALENDAR_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"La Fase 3 requiere la salida de {PREPROCESSING_VERSION}; faltan: {missing}"
        )
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df[TARGET_DATE] = pd.to_datetime(df[TARGET_DATE], errors="coerce")
    if df[DATE].isna().any() or df[TARGET_DATE].isna().any():
        raise ValueError("Se detectaron fechas no válidas.")
    df = df.sort_values(DATE).reset_index(drop=True)
    if df[DATE].duplicated().any():
        raise ValueError("Se detectaron fechas de origen duplicadas.")
    if not df[DATE].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("El dataset no conserva un calendario diario continuo.")
    if not (df[TARGET_DATE] == df[DATE] + pd.Timedelta(days=1)).all():
        raise ValueError("fecha_target no corresponde al día calendario siguiente.")

    mandatory = [PRICE, VOLUME, *CALENDAR_COLUMNS]
    if not np.isfinite(df[mandatory].to_numpy(dtype=float)).all():
        raise ValueError("Las entradas imputadas y el calendario deben ser finitos.")
    if (df[PRICE] <= 0).any() or (df[VOLUME] < 0).any():
        raise ValueError("Precio de entrada no positivo o abastecimiento negativo.")
    target_array = df[TARGET].to_numpy(dtype=float)
    if np.isinf(target_array).any() or (target_array[np.isfinite(target_array)] <= 0).any():
        raise ValueError("Objetivos registrados con valores inválidos.")

    def validar_bandera(column: str, expected: np.ndarray) -> None:
        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError(f"La bandera {column} debe contener solamente 0 o 1.")
        if not np.array_equal(values.astype(bool), expected):
            raise ValueError(f"Inconsistencia de auditoría en {column}.")

    validar_bandera(TARGET_OBSERVED, np.isfinite(target_array))
    for original, entrada, flag, antiguedad in (
        (PRICE_ORIGINAL, PRICE, PRICE_IMPUTED, MISSING_DAYS_PRICE),
        (VOLUME_ORIGINAL, VOLUME, VOLUME_IMPUTED, MISSING_DAYS_VOLUME),
    ):
        raw = df[original].to_numpy(dtype=float)
        if np.isinf(raw).any():
            raise ValueError(f"Valor infinito en {original}.")
        validar_bandera(flag, np.isnan(raw))
        observed = np.isfinite(raw)
        entered = df[entrada].to_numpy(dtype=float)
        if not np.array_equal(raw[observed], entered[observed]):
            raise ValueError(f"La Fase 2 modificó observaciones reales de {original}.")
        expected_age = (
            df[DATE] - df[DATE].where(df[original].notna()).ffill()
        ).dt.days.to_numpy(dtype=float)
        if not np.array_equal(expected_age, df[antiguedad].to_numpy(dtype=float)):
            raise ValueError(f"Antigüedad temporal inconsistente en {antiguedad}.")

    original_price = df[PRICE_ORIGINAL].to_numpy(dtype=float)
    if len(df) > 1 and not np.array_equal(
        target_array[:-1], original_price[1:], equal_nan=True
    ):
        raise ValueError(
            "El objetivo t+1 no coincide con el precio ORIGINAL del siguiente día."
        )
    if not np.array_equal(
        df[PRICE].to_numpy(dtype=float),
        df[PRICE_ORIGINAL].ffill().to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("La entrada de precio no respeta LOCF causal.")
    estado_ewma = df[VOLUME_ORIGINAL].ewm(
        alpha=float(protocolo["alpha_ewma"]), adjust=False, ignore_na=True
    ).mean()
    expected_volume = df[VOLUME_ORIGINAL].where(
        df[VOLUME_ORIGINAL].notna(), estado_ewma
    ).to_numpy(dtype=float)
    if not np.allclose(df[VOLUME].to_numpy(dtype=float), expected_volume,
                       atol=1e-8, rtol=1e-9, equal_nan=True):
        raise ValueError(f"La entrada de abastecimiento no respeta EWMA causal alpha={protocolo['alpha_ewma']}.")
    return df


def normalizar_rutas_artefactos(obj):
    """Guardar rutas relativas al proyecto, no a un perfil de Windows."""
    if isinstance(obj, dict):
        return {k: normalizar_rutas_artefactos(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [normalizar_rutas_artefactos(v) for v in obj]
    if isinstance(obj, str):
        root=str(ROOT.resolve())
        if obj.startswith(root + os.sep):
            return Path(obj).resolve().relative_to(ROOT.resolve()).as_posix()
    return obj


def describir_variables_por_modelo(df: pd.DataFrame) -> None:
    """Distingue explícitamente SARIMAX de ML/DL en covariables calendario."""
    TABLE_DIR.mkdir(parents=True,exist_ok=True)
    r=[]
    for cfg in CONFIGURACIONES:
        for model in ("Naive","SeasonalNaive_7","SARIMAX","XGBoost","LSTM","TCN","LSTM-XGBoost"):
            if model=="SARIMAX":
                variables="precio histórico + abastecimiento contemporáneo" if cfg==CONFIG_HISTORICO_ABASTECIMIENTO else "precio histórico solamente"
                calendario="No, no tiene senos/cosenos explícitos"
            elif model in ("Naive","SeasonalNaive_7"):
                variables="precio histórico";calendario="No"
            else:
                variables="precio, calendario" + (", abastecimiento" if cfg==CONFIG_HISTORICO_ABASTECIMIENTO else "")
                calendario="Sí"
            r.append({"modelo":model,"configuracion":etiqueta_configuracion(cfg),"grupos_de_entradas":variables,"calendario_ciclico":calendario})
    pd.DataFrame(r).to_csv(TABLE_DIR/"Tabla_3c_variables_por_modelo.csv",index=False,encoding="utf-8-sig")


def comprobar_muestras_comunes(df:pd.DataFrame,folds) -> None:
    """Verificar objetivos compartidos por todos los modelos, sin usar test."""
    registros=[]
    for k,(a,b) in enumerate(folds,1):
        base=set(int(j) for j in b if np.isfinite(df.loc[j,TARGET]))
        for config in CONFIGURACIONES:
            x,y,rows,_=preparar_tabular(df,b,config)
            for window in (7,14,30,45,60):
                xx,yy,ids=construir_ventanas(df,b,window,config,usar_calendario=True)
                faltan=base-set(map(int,ids))
                registros.append({"fold":k,"configuracion":etiqueta_configuracion(config),
                                  "ventana":window,"n_objetivos_observados":len(base),
                                  "n_evaluables_ventana":len(ids),"n_no_evaluables":len(faltan)})
    pd.DataFrame(registros).to_csv(TABLE_DIR/"Tabla_3d_cobertura_de_fechas_cv.csv",index=False,encoding="utf-8-sig")
    if any(v["n_no_evaluables"] for v in registros):
        raise AssertionError("Ventanas DL no cubren las mismas fechas evaluables por fold; revisar comparación")


def validar_sensibilidad_21_dias(df:pd.DataFrame,folds,params:dict) -> None:
    """Experimento auxiliar XGB: conservar/excluir filas afectadas en entrenamiento.

    Fecha de evaluación idéntica en ambas políticas; excluye test completo.
    No altera el modelo principal ni selecciona hiperparámetros.
    """
    DIAGNOSTICS_DIR.mkdir(parents=True,exist_ok=True)
    df=df.iloc[:int(folds[-1][1][-1])+1].copy()
    missing=df[PRICE_ORIGINAL].isna().to_numpy() & df[VOLUME_ORIGINAL].isna().to_numpy()
    marker=np.zeros(len(df),bool)
    change=np.diff(np.r_[False,missing,False].astype(int))
    for first,last in zip(np.where(change==1)[0],np.where(change==-1)[0]):
        if last-first>=21:marker[first:last]=True
    if not marker.any():
        raise RuntimeError("No hay tramo de 21 días: se requiere revisar manualmente la sensibilidad")
    affected=pd.Series(marker).rolling(30,min_periods=1).max().fillna(0).to_numpy(dtype=bool)
    entries=[]
    for k,(train,val) in enumerate(folds,1):
        val=np.asarray([i for i in val if not affected[i]],dtype=int)
        for name,keep in (("A_conservar",train),("B_excluir",np.asarray([i for i in train if not affected[i]],dtype=int))):
            Xtr,ytr,_,_=preparar_tabular(df,keep,CONFIG_HISTORICO_ABASTECIMIENTO)
            Xv,yv,_,_=preparar_tabular(df,val,CONFIG_HISTORICO_ABASTECIMIENTO)
            if len(Xtr)<50 or len(yv)<5:raise RuntimeError("Sensibilidad 21d sin muestras suficientes")
            model=crear_xgboost(params);model.fit(Xtr,ytr)
            score=calcular_metricas(yv,model.predict(Xv))
            entries.append({"fold":k,"politica":name,"n_train":len(Xtr),"n_val_comun":len(yv),**score})
    pd.DataFrame(entries).to_csv(DIAGNOSTICS_DIR/"Tabla_3e_sensibilidad_21_dias_xgboost.csv",index=False,encoding="utf-8-sig")
    print("Diagnóstico 21 días: conjunto de validación emparejado por fold, prueba no utilizada.")


def sensibilidad_alpha_en_pronostico(df:pd.DataFrame,folds,params:dict) -> None:
    """Sensibilidad condicional del sMAPE de XGBoost con sus parámetros fijos.

    No es una optimización confirmatoria y no cambia el alpha principal.
    """
    DIAGNOSTICS_DIR.mkdir(parents=True,exist_ok=True)
    base=df.iloc[:int(folds[-1][1][-1])+1].copy()
    volume_obs=base[VOLUME_ORIGINAL]
    records=[]
    for alpha in (0.10,0.20,0.30,0.40,0.50):
        d=base.copy()
        smoothed=volume_obs.ewm(alpha=alpha,adjust=False,ignore_na=True).mean()
        d[VOLUME]=volume_obs.where(volume_obs.notna(),smoothed)
        for lag in (1,7,14,30):
            d[f"abastecimiento_lag_{lag}"]=d[VOLUME].shift(lag-1)
        for k,(train,val) in enumerate(folds,1):
            Xtr,ytr,_,_=preparar_tabular(d,train,CONFIG_HISTORICO_ABASTECIMIENTO)
            Xv,yv,_,_=preparar_tabular(d,val,CONFIG_HISTORICO_ABASTECIMIENTO)
            model=crear_xgboost(params);model.fit(Xtr,ytr)
            metrics=calcular_metricas(yv,model.predict(Xv))
            records.append({"alpha":alpha,"fold":k,"n_validacion":len(yv),**metrics})
    table=pd.DataFrame(records)
    table.to_csv(DIAGNOSTICS_DIR/"Tabla_3f_alpha_sMAPE_cv_xgboost.csv",index=False,encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(7,4))
    for k,g in table.groupby("fold"):
        ax.plot(g.alpha,g.smape,marker="o",alpha=.35,label=f"Fold {k}")
    avg=table.groupby("alpha").smape.mean()
    ax.plot(avg.index,avg.values,lw=2,color="black",marker="s",label="Media")
    ax.set(xlabel="Alpha EWMA",ylabel="sMAPE validación (%)",
           title="Sensibilidad exploratoria de alpha - XGBoost, sólo desarrollo")
    ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(FIGURE_DIR/"Figura_3_sensibilidad_alpha_cv.png",dpi=300,bbox_inches="tight");plt.close(fig)
    print("Sensibilidad predictiva de alpha: diagnóstico posterior, sin cambiar el valor seleccionado por Fase 2.")


def filtrar_pares_observados(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """El objetivo ausente se excluye; una predicción inválida NO se oculta."""
    true = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(true) != len(predicted):
        raise ValueError("No coincide la longitud de observaciones y pronósticos.")
    valid = np.isfinite(true)
    if not valid.any():
        raise ValueError("No hay precios reales para calcular métricas.")
    if not np.isfinite(predicted[valid]).all():
        raise ValueError("Un modelo produjo un pronóstico no finito para un objetivo observado.")
    return true[valid], predicted[valid]


def guardar_auditoria_temporal(
    df: pd.DataFrame,
    implementation_indices: np.ndarray,
    test_indices: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Auditoría sin consultar métricas o hiperparámetros de la prueba final."""
    rows = []
    blocks = [("desarrollo_80", implementation_indices), ("prueba_20", test_indices)]
    for f, (train, val) in enumerate(folds, 1):
        blocks.extend(((f"fold_{f}_train", train), (f"fold_{f}_val", val)))
    for name, idx in blocks:
        block = df.loc[idx]
        rows.append({
            "bloque": name,
            "inicio_origen": block[DATE].iloc[0],
            "fin_origen": block[DATE].iloc[-1],
            "fechas_calendario": len(idx),
            "objetivos_observados": int(block[TARGET_OBSERVED].sum()),
            "objetivos_ausentes": int((block[TARGET_OBSERVED] == 0).sum()),
            "precios_entrada_imputados": int(block[PRICE_IMPUTED].sum()),
            "abastecimientos_entrada_imputados": int(block[VOLUME_IMPUTED].sum()),
            "max_dias_sin_precio": int(block[MISSING_DAYS_PRICE].max()),
            "max_dias_sin_abastecimiento": int(block[MISSING_DAYS_VOLUME].max()),
        })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "auditoria_disponibilidad_fase3.csv",
        index=False, encoding="utf-8-sig", date_format="%Y-%m-%d",
    )


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
        column
        for column in df.columns
        if column.startswith("abastecimiento_lag_")
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


def separar_implementacion_prueba(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cut = int(math.floor(len(df) * IMPLEMENTATION_RATIO))
    if cut <= 0 or cut >= len(df):
        raise ValueError("No es posible construir la partición cronológica 80/20.")

    implementation_indices = np.arange(0, cut, dtype=int)
    test_indices = np.arange(cut, len(df), dtype=int)

    metadata = {
        "proporcion_implementacion": IMPLEMENTATION_RATIO,
        "proporcion_prueba": 1.0 - IMPLEMENTATION_RATIO,
        "n_total": int(len(df)),
        "n_implementacion": int(len(implementation_indices)),
        "n_prueba": int(len(test_indices)),
        "n_objetivos_observados_desarrollo": int(df.loc[implementation_indices, TARGET_OBSERVED].sum()),
        "n_objetivos_ausentes_desarrollo": int((df.loc[implementation_indices, TARGET_OBSERVED] == 0).sum()),
        "n_objetivos_observados_prueba": int(df.loc[test_indices, TARGET_OBSERVED].sum()),
        "n_objetivos_ausentes_prueba": int((df.loc[test_indices, TARGET_OBSERVED] == 0).sum()),
        "protocolo_preprocesamiento": PREPROCESSING_VERSION,
        "origen_implementacion_inicio": str(df.loc[implementation_indices[0], DATE].date()),
        "origen_implementacion_fin": str(df.loc[implementation_indices[-1], DATE].date()),
        "objetivo_implementacion_inicio": str(df.loc[implementation_indices[0], TARGET_DATE].date()),
        "objetivo_implementacion_fin": str(df.loc[implementation_indices[-1], TARGET_DATE].date()),
        "origen_prueba_inicio": str(df.loc[test_indices[0], DATE].date()),
        "origen_prueba_fin": str(df.loc[test_indices[-1], DATE].date()),
        "objetivo_prueba_inicio": str(df.loc[test_indices[0], TARGET_DATE].date()),
        "objetivo_prueba_fin": str(df.loc[test_indices[-1], TARGET_DATE].date()),
    }

    return implementation_indices, test_indices, metadata


def construir_folds_temporales(
    train_indices: np.ndarray,
    n_splits: int = N_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for train_pos, val_pos in splitter.split(train_indices):
        folds.append(
            (
                train_indices[train_pos],
                train_indices[val_pos],
            )
        )

    return folds


def guardar_folds(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    rows: list[dict] = []
    for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
        rows.append(
            {
                "fold": fold_number,
                "n_entrenamiento": len(train_idx),
                "n_validacion": len(val_idx),
                "n_entrenamiento_observados": int(df.loc[train_idx, TARGET_OBSERVED].sum()),
                "n_validacion_observados": int(df.loc[val_idx, TARGET_OBSERVED].sum()),
                "n_validacion_ausentes": int((df.loc[val_idx, TARGET_OBSERVED] == 0).sum()),
                "entrenamiento_inicio": df.loc[train_idx[0], DATE],
                "entrenamiento_fin": df.loc[train_idx[-1], DATE],
                "validacion_inicio": df.loc[val_idx[0], DATE],
                "validacion_fin": df.loc[val_idx[-1], DATE],
                "objetivo_validacion_inicio": df.loc[val_idx[0], TARGET_DATE],
                "objetivo_validacion_fin": df.loc[val_idx[-1], TARGET_DATE],
            }
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "folds_temporales.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )


def evaluar_baselines(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict] = []
    imprimir_subtitulo("BASELINES - VALIDACIÓN TEMPORAL")

    for fold_number, (_, val_idx) in enumerate(folds, start=1):
        y_true = df.loc[val_idx, TARGET].to_numpy(dtype=float)
        naive_pred = df.loc[val_idx, PRICE].to_numpy(dtype=float)

        seasonal_column = "precio_lag_7"
        if seasonal_column not in df.columns:
            raise ValueError(
                "La Fase 2 no contiene precio_lag_7 para Seasonal Naïve."
            )
        seasonal_pred = df.loc[val_idx, seasonal_column].to_numpy(dtype=float)

        for model_name, predictions in (
            ("Naive", naive_pred),
            ("SeasonalNaive_7", seasonal_pred),
        ):
            y_eval, predictions_eval = filtrar_pares_observados(y_true, predictions)
            metrics = calcular_metricas(y_eval, predictions_eval)
            print(
                f"  Fold {fold_number}/{len(folds)} | {model_name:<18} "
                f"sMAPE={metrics['smape']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}",
                flush=True,
            )
            rows.append(
                {
                    "modelo": model_name,
                    "configuracion": etiqueta_configuracion(CONFIG_HISTORICO),
                    "fold": fold_number,
                    **metrics,
                    "n": len(y_eval),
                    "n_objetivos_ausentes": int((~np.isfinite(y_true)).sum()),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "baselines_validacion.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def preparar_tabular(
    df: pd.DataFrame,
    indices: np.ndarray,
    configuracion: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = obtener_columnas_tabulares(df, configuracion)
    X = df.loc[indices, columns].to_numpy(dtype=float)
    y = df.loc[indices, TARGET].to_numpy(dtype=float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

    return X[valid], y[valid], indices[valid], columns


def construir_ventanas(
    df: pd.DataFrame,
    indices: np.ndarray,
    window: int,
    configuracion: str,
    usar_calendario: bool = True,
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

        if not np.isfinite(values).all() or not np.isfinite(target[i]):
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


def ajustar_escaladores_secuenciales(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[MinMaxScaler, MinMaxScaler]:
    n_features = X_train.shape[2]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(X_train.reshape(-1, n_features))
    y_scaler.fit(y_train.reshape(-1, 1))
    return x_scaler, y_scaler


def escalar_secuencias(
    X: np.ndarray,
    y: np.ndarray,
    x_scaler: MinMaxScaler,
    y_scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, window, n_features = X.shape
    X_scaled = x_scaler.transform(X.reshape(-1, n_features)).reshape(
        n_samples,
        window,
        n_features,
    )
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)


def crear_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def entrenar_red_con_validacion(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, int, np.ndarray, dict[str, list[float]]]:
    model = model.to(DEVICE)
    train_loader = crear_loader(X_train, y_train, batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_val_tensor = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_tensor = torch.from_numpy(y_val).float().to(DEVICE)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    patience_count = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            batch_n = int(len(y_batch))
            train_loss_sum += float(loss.item()) * batch_n
            train_count += batch_n

        train_loss = train_loss_sum / max(train_count, 1)
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = float(criterion(val_predictions, y_val_tensor).item())

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_loss - 1e-10:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor).detach().cpu().numpy()

    return model, best_epoch, predictions, history


def entrenar_red_final(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    historial_archivo: Path | None = None,
    titulo: str | None = None,
) -> nn.Module:
    """Reajuste con épocas fijas, sin early stopping sobre prueba.

    Registrar el MSE medio de cada época no cambia la función de pérdida, las
    muestras, el orden de lotes ni los pesos. No se calcula pérdida de prueba.
    """
    model = model.to(DEVICE)
    loader = crear_loader(X_train, y_train, batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        acumulado, cantidad = 0., 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            acumulado += float(loss.item()) * len(y_batch)
            cantidad += len(y_batch)
        if historial_archivo is not None:
            history.append({"epoca": epoch, "mse_entrenamiento_escalado": acumulado / cantidad})
    if historial_archivo is not None:
        historial_archivo.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(historial_archivo, index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(7.3, 4.2))
        ax.plot([h["epoca"] for h in history], [h["mse_entrenamiento_escalado"] for h in history])
        ax.set(title=titulo or "Reajuste del modelo final", xlabel="Época", ylabel="MSE de entrenamiento (objetivo escalado)")
        ax.grid(alpha=.18); fig.tight_layout()
        ruta = FIGURE_DIR / f"{historial_archivo.stem}.png"
        fig.savefig(ruta, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"Curva de reajuste registrada: {ruta}", flush=True)
    return model


def predecir_red(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        predictions = model(tensor).detach().cpu().numpy()
    return predictions


def generar_candidatos_sarimax() -> list[SarimaxCandidate]:
    candidates: set[
        tuple[tuple[int, int, int], tuple[int, int, int, int]]
    ] = set()

    for p, d, q in product(range(3), range(2), range(3)):
        if p + q <= SARIMAX_MAX_TOTAL_ARMA_ORDER:
            candidates.add(((p, d, q), (0, 0, 0, 0)))

        for P, D, Q in product(range(2), range(2), range(2)):
            if (P, D, Q) == (0, 0, 0):
                continue
            if p + q + P + Q <= SARIMAX_MAX_TOTAL_ARMA_ORDER:
                candidates.add(
                    (
                        (p, d, q),
                        (P, D, Q, SARIMAX_SEASONAL_PERIOD),
                    )
                )

    return [
        SarimaxCandidate(order=order, seasonal_order=seasonal_order)
        for order, seasonal_order in sorted(candidates)
    ]


def preparar_exog_sarimax(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    other_idx: np.ndarray | None,
    configuracion: str,
) -> tuple[np.ndarray | None, np.ndarray | None, StandardScaler | None]:
    if configuracion == CONFIG_HISTORICO:
        return None, None, None

    exog_column = "abastecimiento_lag_1"
    if exog_column not in df.columns:
        raise ValueError(
            "La configuración Histórico + abastecimiento requiere abastecimiento_lag_1 generado en la Fase 2."
        )

    scaler = StandardScaler()
    train_exog = df.loc[train_idx, [exog_column]].to_numpy(dtype=float)
    if not np.isfinite(train_exog).all():
        raise ValueError("SARIMAX encontró regresores exógenos no finitos en entrenamiento.")
    train_scaled = scaler.fit_transform(train_exog)

    if other_idx is None:
        return train_scaled, None, scaler

    other_exog = df.loc[other_idx, [exog_column]].to_numpy(dtype=float)
    if not np.isfinite(other_exog).all():
        raise ValueError("SARIMAX encontró regresores exógenos no finitos fuera de entrenamiento.")
    other_scaled = scaler.transform(other_exog)
    return train_scaled, other_scaled, scaler


def ajustar_sarimax(
    y_train: np.ndarray,
    exog_train: np.ndarray | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
):
    methods = (
        ("lbfgs", SARIMAX_MAXITER_LBFGS),
        ("powell", SARIMAX_MAXITER_POWELL),
    )

    last_error: Exception | None = None

    for method, maxiter in methods:
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", ConvergenceWarning)
                model = SARIMAX(
                    endog=y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(
                    disp=False,
                    method=method,
                    maxiter=maxiter,
                )

            converged = bool(result.mle_retvals.get("converged", True))
            if converged and np.isfinite(result.aic):
                return result, method
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        return None, None
    return None, None


def preseleccionar_sarimax(
    df: pd.DataFrame,
    first_train_idx: np.ndarray,
    configuracion: str,
) -> tuple[list[SarimaxCandidate], pd.DataFrame]:
    candidates = generar_candidatos_sarimax()
    y_train = df.loc[first_train_idx, TARGET].to_numpy(dtype=float)
    exog_train, _, _ = preparar_exog_sarimax(
        df,
        first_train_idx,
        None,
        configuracion,
    )

    rows: list[dict] = []
    valid_candidates: list[SarimaxCandidate] = []

    print(
        f"Preselección SARIMAX-{etiqueta_configuracion(configuracion)}: {len(candidates)} especificaciones candidatas por AIC.",
        flush=True,
    )

    for candidate_number, candidate in enumerate(candidates, start=1):
        print(
            f"  Preselección {candidate_number}/{len(candidates)} | "
            f"order={candidate.order} | seasonal={candidate.seasonal_order}",
            flush=True,
        )
        result, optimizer = ajustar_sarimax(
            y_train,
            exog_train,
            candidate.order,
            candidate.seasonal_order,
        )
        converged = result is not None
        aic = float(result.aic) if converged else None
        evaluated = SarimaxCandidate(
            order=candidate.order,
            seasonal_order=candidate.seasonal_order,
            aic=aic,
            convergio=converged,
            optimizador=optimizer,
        )
        rows.append(asdict(evaluated))
        if converged:
            valid_candidates.append(evaluated)
            print(
                f"    Convergió con {optimizer} | AIC={aic:.4f}",
                flush=True,
            )
        else:
            print("    No convergió | candidato descartado", flush=True)

    if not valid_candidates:
        raise RuntimeError(
            f"Ninguna especificación SARIMAX convergió en la preselección de {etiqueta_configuracion(configuracion)}."
        )

    valid_candidates.sort(key=lambda item: float(item.aic))
    selected = valid_candidates[:SARIMAX_TOP_K]
    print(
        f"Preselección SARIMAX-{etiqueta_configuracion(configuracion)} completada: "
        f"{len(valid_candidates)} convergieron y {len(selected)} pasan a validación temporal.",
        flush=True,
    )
    return selected, pd.DataFrame(rows)


def pronosticar_sarimax_un_paso(
    result,
    y_val: np.ndarray,
    exog_val: np.ndarray | None,
) -> np.ndarray:
    predictions: list[float] = []
    current = result

    for position, actual in enumerate(y_val):
        exog_step = None
        if exog_val is not None:
            exog_step = exog_val[position : position + 1]

        forecast = current.forecast(steps=1, exog=exog_step)
        predictions.append(float(np.asarray(forecast).ravel()[0]))
        current = current.append(
            endog=np.asarray([actual], dtype=float),
            exog=exog_step,
            refit=False,
        )

    return np.asarray(predictions, dtype=float)


def seleccionar_sarimax(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    configuracion: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    candidates, prescreen = preseleccionar_sarimax(
        df,
        folds[0][0],
        configuracion,
    )
    fold_rows: list[dict] = []
    candidate_summary: list[dict] = []
    imprimir_subtitulo(f"SARIMAX - {etiqueta_configuracion(configuracion)}")

    for candidate_id, candidate in enumerate(candidates, start=1):
        print(
            f"Candidato {candidate_id}/{len(candidates)} | "
            f"order={candidate.order} | seasonal={candidate.seasonal_order}",
            flush=True,
        )
        fold_smapes: list[float] = []
        valid_all = True

        for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
            y_train = df.loc[train_idx, TARGET].to_numpy(dtype=float)
            y_val = df.loc[val_idx, TARGET].to_numpy(dtype=float)
            exog_train, exog_val, _ = preparar_exog_sarimax(
                df,
                train_idx,
                val_idx,
                configuracion,
            )
            result, optimizer = ajustar_sarimax(
                y_train,
                exog_train,
                candidate.order,
                candidate.seasonal_order,
            )

            if result is None:
                valid_all = False
                print(
                    f"  Fold {fold_number}/{len(folds)} | no convergió | candidato descartado",
                    flush=True,
                )
                fold_rows.append(
                    {
                        "modelo": "SARIMAX",
                        "configuracion": etiqueta_configuracion(configuracion),
                        "candidato": candidate_id,
                        "order": str(candidate.order),
                        "seasonal_order": str(candidate.seasonal_order),
                        "fold": fold_number,
                        "convergio": False,
                        "optimizador": None,
                        "smape": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "n": len(val_idx),
                    }
                )
                break

            predictions = pronosticar_sarimax_un_paso(
                result,
                y_val,
                exog_val,
            )
            observed_y, observed_predictions = filtrar_pares_observados(y_val, predictions)
            metrics = calcular_metricas(observed_y, observed_predictions)
            fold_smapes.append(metrics["smape"])
            print(
                f"  Fold {fold_number}/{len(folds)} | {optimizer} | "
                f"sMAPE={metrics['smape']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}",
                flush=True,
            )
            fold_rows.append(
                {
                    "modelo": "SARIMAX",
                    "configuracion": etiqueta_configuracion(configuracion),
                    "candidato": candidate_id,
                    "order": str(candidate.order),
                    "seasonal_order": str(candidate.seasonal_order),
                    "fold": fold_number,
                    "convergio": True,
                    "optimizador": optimizer,
                    **metrics,
                    "n": len(observed_y),
                    "n_objetivos_ausentes": int((~np.isfinite(y_val)).sum()),
                }
            )

        candidate_mean = (
            float(np.mean(fold_smapes))
            if valid_all and len(fold_smapes) == len(folds)
            else np.inf
        )
        if np.isfinite(candidate_mean):
            print(f"  sMAPE promedio del candidato: {candidate_mean:.4f}", flush=True)

        candidate_summary.append(
            {
                "candidato": candidate_id,
                "order": candidate.order,
                "seasonal_order": candidate.seasonal_order,
                "valido_cinco_folds": valid_all and len(fold_smapes) == len(folds),
                "smape_promedio": candidate_mean,
            }
        )

    valid_summaries = [
        row for row in candidate_summary if row["valido_cinco_folds"]
    ]
    if not valid_summaries:
        raise RuntimeError(
            f"Ningún candidato SARIMAX de {etiqueta_configuracion(configuracion)} convergió en los cinco folds."
        )

    best = min(valid_summaries, key=lambda row: row["smape_promedio"])
    print(
        f"Mejor SARIMAX-{etiqueta_configuracion(configuracion)}: order={best['order']} | "
        f"seasonal={best['seasonal_order']} | sMAPE CV={best['smape_promedio']:.4f}",
        flush=True,
    )
    return best, prescreen, pd.DataFrame(fold_rows)


def entrenar_sarimax_final(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    configuracion: str,
    best: dict,
) -> dict:
    print(f"Reajustando SARIMAX-{etiqueta_configuracion(configuracion)} con todo el conjunto de implementación (80 %)...", flush=True)
    y_train = df.loc[train_indices, TARGET].to_numpy(dtype=float)
    exog_train, _, scaler = preparar_exog_sarimax(
        df,
        train_indices,
        None,
        configuracion,
    )
    result, optimizer = ajustar_sarimax(
        y_train,
        exog_train,
        tuple(best["order"]),
        tuple(best["seasonal_order"]),
    )
    if result is None:
        raise RuntimeError(
            f"El SARIMAX seleccionado para {etiqueta_configuracion(configuracion)} no convergió al reajustarse con todo el 80 %."
        )

    model_path = MODEL_DIR / f"sarimax_{slug_configuracion(configuracion)}.pkl"
    result.save(str(model_path))
    print(f"SARIMAX-{etiqueta_configuracion(configuracion)} reajustado y guardado.", flush=True)

    scaler_path = None
    if scaler is not None:
        scaler_path = MODEL_DIR / f"sarimax_{slug_configuracion(configuracion)}_exog_scaler.joblib"
        joblib.dump(scaler, scaler_path)

    return {
        "modelo": str(model_path),
        "escalador_exog": str(scaler_path) if scaler_path else None,
        "optimizador": optimizer,
    }


def crear_xgboost(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def sugerir_parametros_xgboost(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-3,
            1e-1,
            log=True,
        ),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            500,
            step=50,
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-3,
            1.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            10.0,
            log=True,
        ),
    }



def seleccionar_xgboost(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    configuracion: str,
) -> tuple[dict, optuna.Study, pd.DataFrame, dict]:
    fold_records: dict[int, list[dict]] = {}
    trial_histories: dict[int, list[dict]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = sugerir_parametros_xgboost(trial)
        records: list[dict] = []
        histories: list[dict] = []
        scores: list[float] = []
        print(
            f"Trial {trial.number + 1}/{XGB_TRIALS} | {formatear_parametros(params)}",
            flush=True,
        )

        for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
            X_train, y_train, _, _ = preparar_tabular(
                df,
                train_idx,
                configuracion,
            )
            X_val, y_val, _, _ = preparar_tabular(
                df,
                val_idx,
                configuracion,
            )

            model = crear_xgboost(params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
            predictions = model.predict(X_val)
            metrics = calcular_metricas(y_val, predictions)
            scores.append(metrics["smape"])

            evals = model.evals_result()
            train_rmse = [float(v) for v in evals["validation_0"]["rmse"]]
            val_rmse = [float(v) for v in evals["validation_1"]["rmse"]]
            histories.append(
                {
                    "fold": fold_number,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                }
            )

            print(
                f"  Fold {fold_number}/{len(folds)} | "
                f"sMAPE={metrics['smape']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}",
                flush=True,
            )
            records.append(
                {
                    "modelo": "XGBoost",
                    "configuracion": etiqueta_configuracion(configuracion),
                    "trial": trial.number,
                    "fold": fold_number,
                    **metrics,
                    "n": len(y_val),
                }
            )
            trial.report(float(np.mean(scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fold_records[trial.number] = records
        trial_histories[trial.number] = histories
        trial_mean = float(np.mean(scores))
        print(f"  sMAPE promedio del trial: {trial_mean:.4f}", flush=True)
        return trial_mean

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"xgboost_{slug_configuracion(configuracion)}",
    )
    imprimir_subtitulo(f"XGBOOST - {etiqueta_configuracion(configuracion)}")
    study.optimize(objective, n_trials=XGB_TRIALS, show_progress_bar=False)
    print(
        f"Mejor XGBoost-{etiqueta_configuracion(configuracion)}: sMAPE CV={study.best_value:.4f} | "
        f"{formatear_parametros(study.best_params)}",
        flush=True,
    )

    best_trial = study.best_trial.number
    if best_trial not in fold_records or best_trial not in trial_histories:
        raise RuntimeError(
            f"No se conservaron resultados completos del mejor trial XGBoost-{etiqueta_configuracion(configuracion)}."
        )

    best_fold5_history = trial_histories[best_trial][-1]
    return (
        dict(study.best_params),
        study,
        pd.DataFrame(fold_records[best_trial]),
        best_fold5_history,
    )

def entrenar_xgboost_final(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    configuracion: str,
    best_params: dict,
) -> dict:
    print(f"Reajustando XGBoost-{etiqueta_configuracion(configuracion)} con todo el conjunto de implementación (80 %)...", flush=True)
    X_train, y_train, rows, columns = preparar_tabular(
        df,
        train_indices,
        configuracion,
    )
    model = crear_xgboost(best_params)
    model.fit(X_train, y_train)

    model_path = MODEL_DIR / f"xgboost_{slug_configuracion(configuracion)}.joblib"
    joblib.dump(model, model_path)
    print(f"XGBoost-{etiqueta_configuracion(configuracion)} reajustado y guardado.", flush=True)

    return {
        "modelo": str(model_path),
        "columnas": columns,
        "n_entrenamiento": int(len(rows)),
    }


def sugerir_parametros_lstm(trial: optuna.Trial) -> dict:
    return {
        "window": trial.suggest_categorical("window", list(LSTM_WINDOWS)),
        "hidden_size": trial.suggest_int(
            "hidden_size",
            50,
            150,
            step=10,
        ),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            list(LSTM_BATCH_SIZES),
        ),
    }



def evaluar_lstm_en_folds(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    configuracion: str,
    params: dict,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[dict], list[int], list[dict]]:
    scores: list[float] = []
    records: list[dict] = []
    best_epochs: list[int] = []
    histories: list[dict] = []

    for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"  Entrenando fold {fold_number}/{len(folds)}...", flush=True)
        fijar_semilla(SEED + fold_number)
        X_train, y_train, _ = construir_ventanas(
            df,
            train_idx,
            int(params["window"]),
            configuracion,
        )
        X_val, y_val, _ = construir_ventanas(
            df,
            val_idx,
            int(params["window"]),
            configuracion,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise RuntimeError(
                f"No hay secuencias suficientes para LSTM-{etiqueta_configuracion(configuracion)} en el fold {fold_number}."
            )

        x_scaler, y_scaler = ajustar_escaladores_secuenciales(
            X_train,
            y_train,
        )
        X_train_scaled, y_train_scaled = escalar_secuencias(
            X_train,
            y_train,
            x_scaler,
            y_scaler,
        )
        X_val_scaled, y_val_scaled = escalar_secuencias(
            X_val,
            y_val,
            x_scaler,
            y_scaler,
        )

        model = LSTMRegressor(
            n_features=X_train.shape[2],
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
        )
        model, best_epoch, predictions_scaled, history = entrenar_red_con_validacion(
            model,
            X_train_scaled,
            y_train_scaled,
            X_val_scaled,
            y_val_scaled,
            batch_size=int(params["batch_size"]),
            learning_rate=LSTM_LEARNING_RATE,
            max_epochs=LSTM_MAX_EPOCHS,
            patience=LSTM_PATIENCE,
        )
        predictions = y_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).ravel()
        metrics = calcular_metricas(y_val, predictions)
        scores.append(metrics["smape"])
        best_epochs.append(int(best_epoch))
        histories.append(
            {
                "fold": fold_number,
                "train_loss": [float(v) for v in history["train_loss"]],
                "val_loss": [float(v) for v in history["val_loss"]],
                "best_epoch": int(best_epoch),
            }
        )
        print(
            f"    Fold {fold_number}/{len(folds)} | época óptima={best_epoch} | "
            f"sMAPE={metrics['smape']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}",
            flush=True,
        )
        records.append(
            {
                "modelo": "LSTM",
                "configuracion": etiqueta_configuracion(configuracion),
                "fold": fold_number,
                **metrics,
                "n": len(y_val),
                "epoca_optima": int(best_epoch),
            }
        )

        if trial is not None:
            trial.report(float(np.mean(scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(scores)), records, best_epochs, histories


def seleccionar_lstm(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    configuracion: str,
) -> tuple[dict, int, optuna.Study, pd.DataFrame, dict]:
    trial_records: dict[int, list[dict]] = {}
    trial_epochs: dict[int, list[int]] = {}
    trial_histories: dict[int, list[dict]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = sugerir_parametros_lstm(trial)
        print(
            f"Trial {trial.number + 1}/{LSTM_TRIALS} | {formatear_parametros(params)}",
            flush=True,
        )
        score, records, epochs, histories = evaluar_lstm_en_folds(
            df,
            folds,
            configuracion,
            params,
            trial=trial,
        )
        trial_records[trial.number] = records
        trial_epochs[trial.number] = epochs
        trial_histories[trial.number] = histories
        print(f"  sMAPE promedio del trial: {score:.4f}", flush=True)
        return score

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"lstm_{slug_configuracion(configuracion)}",
    )
    imprimir_subtitulo(f"LSTM - {etiqueta_configuracion(configuracion)}")
    study.optimize(objective, n_trials=LSTM_TRIALS, show_progress_bar=False)
    print(
        f"Mejor LSTM-{etiqueta_configuracion(configuracion)}: sMAPE CV={study.best_value:.4f} | "
        f"{formatear_parametros(study.best_params)}",
        flush=True,
    )

    best_trial = study.best_trial.number
    if (
        best_trial not in trial_records
        or best_trial not in trial_epochs
        or best_trial not in trial_histories
    ):
        raise RuntimeError(
            f"No se conservaron resultados completos del mejor trial LSTM-{etiqueta_configuracion(configuracion)}."
        )

    best_epochs = trial_epochs[best_trial]
    final_epochs = max(1, int(round(median(best_epochs))))

    rows = []
    for h in trial_histories[best_trial]:
        fold_number = int(h["fold"])
        for epoch, (train_loss, val_loss) in enumerate(
            zip(h["train_loss"], h["val_loss"]),
            start=1,
        ):
            rows.append(
                {
                    "modelo": "LSTM",
                    "configuracion": etiqueta_configuracion(configuracion),
                    "trial": best_trial,
                    "fold": fold_number,
                    "epoca": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "epoca_optima": int(best_epochs[fold_number - 1]),
                }
            )
    pd.DataFrame(rows).to_csv(
        HISTORY_DIR / f"lstm_curvas_{slug_configuracion(configuracion)}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_fold5_history = dict(trial_histories[best_trial][-1])
    best_fold5_history["best_epoch"] = int(best_epochs[-1])

    return (
        dict(study.best_params),
        final_epochs,
        study,
        pd.DataFrame(trial_records[best_trial]),
        best_fold5_history,
    )

def entrenar_lstm_final(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    configuracion: str,
    best_params: dict,
    epochs: int,
) -> dict:
    print(f"Reajustando LSTM-{etiqueta_configuracion(configuracion)} con todo el conjunto de implementación (80 %)...", flush=True)
    fijar_semilla(SEED)
    X_train, y_train, rows = construir_ventanas(
        df,
        train_indices,
        int(best_params["window"]),
        configuracion,
    )
    x_scaler, y_scaler = ajustar_escaladores_secuenciales(
        X_train,
        y_train,
    )
    X_scaled, y_scaled = escalar_secuencias(
        X_train,
        y_train,
        x_scaler,
        y_scaler,
    )

    model = LSTMRegressor(
        n_features=X_train.shape[2],
        hidden_size=int(best_params["hidden_size"]),
        num_layers=int(best_params["num_layers"]),
        dropout=float(best_params["dropout"]),
    )
    model = entrenar_red_final(
        model,
        X_scaled,
        y_scaled,
        batch_size=int(best_params["batch_size"]),
        learning_rate=LSTM_LEARNING_RATE,
        epochs=epochs,
        historial_archivo=HISTORY_DIR / f"reajuste_lstm_{slug_configuracion(configuracion)}.csv",
        titulo=f"LSTM - {etiqueta_configuracion(configuracion)}: reajuste sobre el 80 %",
    )

    model_path = MODEL_DIR / f"lstm_{slug_configuracion(configuracion)}.pt"
    scaler_path = MODEL_DIR / f"lstm_{slug_configuracion(configuracion)}_scalers.joblib"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": best_params,
            "epochs": epochs,
            "n_features": int(X_train.shape[2]),
            "columnas_secuenciales": obtener_nombres_caracteristicas_secuenciales(configuracion),
        },
        model_path,
    )
    joblib.dump(
        {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        scaler_path,
    )
    print(
        f"LSTM-{etiqueta_configuracion(configuracion)} reajustada y guardada | épocas={epochs}.",
        flush=True,
    )

    return {
        "modelo": str(model_path),
        "escaladores": str(scaler_path),
        "epocas": int(epochs),
        "n_entrenamiento": int(len(rows)),
        "columnas_secuenciales": obtener_nombres_caracteristicas_secuenciales(configuracion),
    }


def calcular_campo_receptivo_tcn(n_blocks: int, kernel_size: int) -> int:
    """Campo receptivo para dos convoluciones por bloque residual."""
    return 1 + 2 * (kernel_size - 1) * sum(2 ** level for level in range(n_blocks))


def sugerir_parametros_tcn(trial: optuna.Trial) -> dict:
    params = {
        "window": trial.suggest_categorical("window", list(TCN_WINDOWS)),
        "hidden_channels": trial.suggest_categorical("hidden_channels", list(TCN_CHANNEL_OPTIONS)),
        "n_blocks": trial.suggest_categorical("n_blocks", list(TCN_BLOCK_OPTIONS)),
        "kernel_size": trial.suggest_categorical("kernel_size", list(TCN_KERNEL_OPTIONS)),
        "dropout": trial.suggest_categorical("dropout", list(TCN_DROPOUT_OPTIONS)),
    }
    rf = calcular_campo_receptivo_tcn(int(params["n_blocks"]), int(params["kernel_size"]))
    if rf < int(params["window"]):
        raise optuna.TrialPruned()
    return params


def evaluar_tcn_en_folds(df, folds, configuracion, params, trial=None):
    scores, records, best_epochs, histories = [], [], [], []
    for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
        inicio_fold = datetime.now()
        print(f"  [{inicio_fold:%H:%M:%S}] Entrenando fold {fold_number}/{len(folds)}...", flush=True)
        fijar_semilla(SEED + fold_number)
        X_train, y_train, _ = construir_ventanas(df, train_idx, int(params["window"]), configuracion, usar_calendario=True)
        X_val, y_val, _ = construir_ventanas(df, val_idx, int(params["window"]), configuracion, usar_calendario=True)
        if len(X_train) == 0 or len(X_val) == 0:
            raise RuntimeError(f"No hay secuencias suficientes para TCN-{etiqueta_configuracion(configuracion)} en fold {fold_number}.")
        x_scaler, y_scaler = ajustar_escaladores_secuenciales(X_train, y_train)
        X_train_scaled, y_train_scaled = escalar_secuencias(X_train, y_train, x_scaler, y_scaler)
        X_val_scaled, y_val_scaled = escalar_secuencias(X_val, y_val, x_scaler, y_scaler)
        model = TCNRegressor(
            n_features=X_train.shape[2], hidden_channels=int(params["hidden_channels"]),
            n_blocks=int(params["n_blocks"]), kernel_size=int(params["kernel_size"]),
            dropout=float(params["dropout"]),
        )
        model, best_epoch, pred_scaled, history = entrenar_red_con_validacion(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            batch_size=TCN_BATCH_SIZE, learning_rate=TCN_LEARNING_RATE,
            max_epochs=TCN_MAX_EPOCHS, patience=TCN_PATIENCE,
        )
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
        metrics = calcular_metricas(y_val, pred)
        scores.append(metrics["smape"]); best_epochs.append(int(best_epoch))
        histories.append({"fold": fold_number, "train_loss": history["train_loss"], "val_loss": history["val_loss"]})
        at_limit = int(best_epoch) >= TCN_MAX_EPOCHS - TCN_AT_LIMIT_MARGIN
        fin_fold = datetime.now()
        print(f"    [{fin_fold:%H:%M:%S}] Fold {fold_number}/{len(folds)} | época óptima={best_epoch} | sMAPE={metrics['smape']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | techo={at_limit} | duración={fin_fold-inicio_fold}", flush=True)
        records.append({
            "modelo":"TCN", "configuracion":etiqueta_configuracion(configuracion), "fold":fold_number,
            **metrics, "n":len(y_val), "epoca_optima":int(best_epoch), "epocas_ejecutadas":len(history["val_loss"]),
            "alcanzo_techo":bool(at_limit), "hidden_channels":int(params["hidden_channels"]),
            "n_blocks":int(params["n_blocks"]), "n_capas_convolucionales":2*int(params["n_blocks"]),
            "campo_receptivo":calcular_campo_receptivo_tcn(int(params["n_blocks"]), int(params["kernel_size"])),
        })
        if trial is not None:
            trial.report(float(np.mean(scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), records, best_epochs, histories



def seleccionar_tcn(df, folds, configuracion):
    trial_records, trial_epochs, trial_histories = {}, {}, {}

    def objective(trial):
        params = sugerir_parametros_tcn(trial)
        print(
            f"Trial {trial.number+1}/{TCN_TRIALS} | {formatear_parametros(params)} | "
            f"campo_receptivo={calcular_campo_receptivo_tcn(int(params['n_blocks']), int(params['kernel_size']))}",
            flush=True,
        )
        score, records, epochs, histories = evaluar_tcn_en_folds(
            df, folds, configuracion, params, trial=trial
        )
        trial_records[trial.number] = records
        trial_epochs[trial.number] = epochs
        trial_histories[trial.number] = histories
        print(f"  sMAPE promedio del trial: {score:.4f}", flush=True)
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=f"tcn_{slug_configuracion(configuracion)}",
    )
    imprimir_subtitulo(f"TCN - {etiqueta_configuracion(configuracion)}")
    study.optimize(objective, n_trials=TCN_TRIALS, show_progress_bar=False)
    print(
        f"Mejor TCN-{etiqueta_configuracion(configuracion)}: sMAPE CV={study.best_value:.4f} | "
        f"{formatear_parametros(study.best_params)}",
        flush=True,
    )
    best_trial = study.best_trial.number
    if (
        best_trial not in trial_records
        or best_trial not in trial_epochs
        or best_trial not in trial_histories
    ):
        raise RuntimeError(
            f"No se conservaron resultados completos del mejor trial TCN-{etiqueta_configuracion(configuracion)}."
        )

    best_epochs = trial_epochs[best_trial]
    final_epochs = max(1, int(round(median(best_epochs))))
    print(
        f"Épocas óptimas del mejor trial: {best_epochs} | mediana para reajuste 80 %={final_epochs}",
        flush=True,
    )

    rows = []
    for h in trial_histories[best_trial]:
        f = int(h["fold"])
        for e, (tl, vl) in enumerate(
            zip(h["train_loss"], h["val_loss"]), start=1
        ):
            rows.append(
                {
                    "modelo": "TCN",
                    "configuracion": etiqueta_configuracion(configuracion),
                    "trial": best_trial,
                    "fold": f,
                    "epoca": e,
                    "train_loss": float(tl),
                    "val_loss": float(vl),
                    "epoca_optima": int(best_epochs[f - 1]),
                }
            )
    pd.DataFrame(rows).to_csv(
        HISTORY_DIR / f"tcn_curvas_{slug_configuracion(configuracion)}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    folds_df = pd.DataFrame(trial_records[best_trial])
    folds_df["epocas_reajuste_mediana"] = final_epochs
    best_fold5_history = dict(trial_histories[best_trial][-1])
    best_fold5_history["best_epoch"] = int(best_epochs[-1])
    return (
        dict(study.best_params),
        final_epochs,
        study,
        folds_df,
        best_fold5_history,
    )

def entrenar_tcn_final(df, train_indices, configuracion, best_params, epochs, smape_cv):
    print(f"Reajustando TCN-{etiqueta_configuracion(configuracion)} con todo el conjunto de implementación (80 %) durante {epochs} épocas...", flush=True)
    fijar_semilla(SEED)
    X_train,y_train,rows = construir_ventanas(df,train_indices,int(best_params["window"]),configuracion,usar_calendario=True)
    x_scaler,y_scaler = ajustar_escaladores_secuenciales(X_train,y_train)
    X_scaled,y_scaled = escalar_secuencias(X_train,y_train,x_scaler,y_scaler)
    model = TCNRegressor(n_features=X_train.shape[2], hidden_channels=int(best_params["hidden_channels"]), n_blocks=int(best_params["n_blocks"]), kernel_size=int(best_params["kernel_size"]), dropout=float(best_params["dropout"]))
    model = entrenar_red_final(model,X_scaled,y_scaled,batch_size=TCN_BATCH_SIZE,learning_rate=TCN_LEARNING_RATE,epochs=epochs, historial_archivo=HISTORY_DIR/f"reajuste_tcn_{slug_configuracion(configuracion)}.csv", titulo=f"TCN - {etiqueta_configuracion(configuracion)}: reajuste sobre el 80 %")
    pred_scaled=predecir_red(model,X_scaled); pred=y_scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
    mt=calcular_metricas(y_train,pred); bias=float(np.mean(pred-y_train)); corr=float(np.corrcoef(pred,y_train)[0,1]) if np.std(pred)>0 and np.std(y_train)>0 else float("nan")
    diag={"modelo":"TCN","configuracion":etiqueta_configuracion(configuracion),"epocas_reajuste":int(epochs),"smape_cv_seleccion":float(smape_cv),"smape_implementacion":float(mt["smape"]),"mae_implementacion":float(mt["mae"]),"rmse_implementacion":float(mt["rmse"]),"sesgo_medio_implementacion":bias,"correlacion_implementacion":corr,"media_real_implementacion":float(np.mean(y_train)),"media_predicha_implementacion":float(np.mean(pred)),"hidden_channels":int(best_params["hidden_channels"]),"n_blocks":int(best_params["n_blocks"])}
    pd.DataFrame([diag]).to_csv(OUTPUT_DIR/f"tcn_diagnostico_reajuste_{slug_configuracion(configuracion)}.csv",index=False,encoding="utf-8-sig")
    if mt["smape"] > 2.0*max(float(smape_cv),1e-12):
        print(f"ADVERTENCIA DIAGNÓSTICA TCN-{etiqueta_configuracion(configuracion)}: sMAPE implementación={mt['smape']:.4f} frente a CV={smape_cv:.4f}. El aviso no modifica la selección.",flush=True)
    model_path=MODEL_DIR/f"tcn_{slug_configuracion(configuracion)}.pt"; scaler_path=MODEL_DIR/f"tcn_{slug_configuracion(configuracion)}_scalers.joblib"
    torch.save({"state_dict":model.state_dict(),"params":best_params,"arquitectura_tcn":TCN_ARCHITECTURE_ID,"hidden_channels":int(best_params["hidden_channels"]),"n_blocks":int(best_params["n_blocks"]),"n_conv_layers":2*int(best_params["n_blocks"]),"dilations":[2**i for i in range(int(best_params["n_blocks"]))],"epochs":int(epochs),"max_epochs_validacion":TCN_MAX_EPOCHS,"patience_validacion":TCN_PATIENCE,"usar_calendario":True,"n_features":int(X_train.shape[2]),"columnas_secuenciales":obtener_nombres_caracteristicas_secuenciales(configuracion,usar_calendario=True)},model_path)
    joblib.dump({"x_scaler":x_scaler,"y_scaler":y_scaler},scaler_path)
    print(f"TCN-{etiqueta_configuracion(configuracion)} guardada | canales={best_params['hidden_channels']} | bloques={best_params['n_blocks']} | épocas={epochs} | sMAPE implementación={mt['smape']:.4f} | sesgo={bias:.6f}",flush=True)
    return {"modelo":str(model_path),"escaladores":str(scaler_path),"arquitectura_tcn":TCN_ARCHITECTURE_ID,"hidden_channels":int(best_params["hidden_channels"]),"n_blocks":int(best_params["n_blocks"]),"epocas":int(epochs),"max_epocas_validacion":TCN_MAX_EPOCHS,"patience_validacion":TCN_PATIENCE,"usar_calendario":True,"n_entrenamiento":int(len(rows)),"columnas_secuenciales":obtener_nombres_caracteristicas_secuenciales(configuracion,usar_calendario=True),"diagnostico_reajuste":diag}


def generar_oof_lstm(
    df: pd.DataFrame,
    indices: np.ndarray,
    configuracion: str,
    lstm_params: dict,
    epochs: int,
) -> pd.Series:
    if epochs <= 0:
        raise ValueError("El número de épocas OOF debe ser mayor que cero.")

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    predictions = pd.Series(index=indices, dtype=float)

    print(
        f"Generando predicciones LSTM OOF para {etiqueta_configuracion(configuracion)} sobre {len(indices)} observaciones "
        f"con {epochs} épocas fijas...",
        flush=True,
    )

    for inner_fold, (train_pos, val_pos) in enumerate(
        splitter.split(indices),
        start=1,
    ):
        print(f"  Subfold OOF {inner_fold}/{N_SPLITS}...", flush=True)
        inner_train = indices[train_pos]
        inner_val = indices[val_pos]
        fijar_semilla(SEED + 100 + inner_fold)

        X_train, y_train, _ = construir_ventanas(
            df,
            inner_train,
            int(lstm_params["window"]),
            configuracion,
        )
        X_val, y_val, val_rows = construir_ventanas(
            df,
            inner_val,
            int(lstm_params["window"]),
            configuracion,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise RuntimeError(
                f"No hay secuencias suficientes para generar OOF LSTM-{etiqueta_configuracion(configuracion)} "
                f"en el subfold {inner_fold}."
            )

        x_scaler, y_scaler = ajustar_escaladores_secuenciales(
            X_train,
            y_train,
        )
        X_train_scaled, y_train_scaled = escalar_secuencias(
            X_train,
            y_train,
            x_scaler,
            y_scaler,
        )
        X_val_scaled, _ = escalar_secuencias(
            X_val,
            y_val,
            x_scaler,
            y_scaler,
        )

        model = LSTMRegressor(
            n_features=X_train.shape[2],
            hidden_size=int(lstm_params["hidden_size"]),
            num_layers=int(lstm_params["num_layers"]),
            dropout=float(lstm_params["dropout"]),
        )
        model = entrenar_red_final(
            model,
            X_train_scaled,
            y_train_scaled,
            batch_size=int(lstm_params["batch_size"]),
            learning_rate=LSTM_LEARNING_RATE,
            epochs=epochs,
        )
        pred_scaled = predecir_red(model, X_val_scaled)
        pred = y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()
        if len(np.intersect1d(inner_train, val_rows)):
            raise AssertionError("OOF: el bloque de predicción intersecta su entrenamiento.")
        if not np.isfinite(y_val).all():
            raise AssertionError("OOF generó etiquetas no observadas.")
        predictions.loc[val_rows] = pred
        print(
            f"    Subfold OOF {inner_fold}/{N_SPLITS} completado | "
            f"épocas fijas={epochs} | predicciones={len(val_rows)}",
            flush=True,
        )

    print(f"Predicciones OOF generadas: {predictions.notna().sum()}.", flush=True)
    output = predictions.dropna()
    if not np.isfinite(df.loc[output.index, TARGET].to_numpy(dtype=float)).all():
        raise AssertionError("Los residuos OOF requieren objetivos reales.")
    return output


def entrenar_lstm_y_predecir_bloque(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    configuracion: str,
    lstm_params: dict,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fijar_semilla(SEED + 500)
    X_train, y_train, _ = construir_ventanas(
        df,
        train_idx,
        int(lstm_params["window"]),
        configuracion,
    )
    X_val, y_val, val_rows = construir_ventanas(
        df,
        val_idx,
        int(lstm_params["window"]),
        configuracion,
    )

    x_scaler, y_scaler = ajustar_escaladores_secuenciales(
        X_train,
        y_train,
    )
    X_train_scaled, y_train_scaled = escalar_secuencias(
        X_train,
        y_train,
        x_scaler,
        y_scaler,
    )
    X_val_scaled, _ = escalar_secuencias(
        X_val,
        y_val,
        x_scaler,
        y_scaler,
    )

    model = LSTMRegressor(
        n_features=X_train.shape[2],
        hidden_size=int(lstm_params["hidden_size"]),
        num_layers=int(lstm_params["num_layers"]),
        dropout=float(lstm_params["dropout"]),
    )
    model = entrenar_red_final(
        model,
        X_train_scaled,
        y_train_scaled,
        batch_size=int(lstm_params["batch_size"]),
        learning_rate=LSTM_LEARNING_RATE,
        epochs=epochs,
    )
    pred_scaled = predecir_red(model, X_val_scaled)
    pred = y_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    return pred, y_val, val_rows


def construir_features_residuales(
    df: pd.DataFrame,
    rows: np.ndarray,
    lstm_predictions: np.ndarray,
    configuracion: str,
) -> tuple[np.ndarray, list[str]]:
    columns = obtener_columnas_tabulares(df, configuracion)
    tabular = df.loc[rows, columns].to_numpy(dtype=float)
    predictions = np.asarray(lstm_predictions, dtype=float).reshape(-1, 1)
    X = np.column_stack((predictions, tabular))
    valid = np.isfinite(X).all(axis=1)

    if not valid.all():
        raise ValueError(
            "Se detectaron características no finitas en la etapa residual después de la alineación temporal."
        )

    return X, ["prediccion_lstm"] + columns


def seleccionar_lstm_solo_con_entrenamiento_externo(
    df: pd.DataFrame,
    outer_train: np.ndarray,
    configuracion: str,
    outer_fold: int,
) -> tuple[dict, int]:
    """Optimización anidada: la validación externa NUNCA selecciona la LSTM base."""
    inner_folds = construir_folds_temporales(outer_train)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED + outer_fold),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=f"lstm_hibrido_nested_{slug_configuracion(configuracion)}_fold{outer_fold}",
    )
    epochs_by_trial: dict[int, list[int]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = sugerir_parametros_lstm(trial)
        score, _, best_epochs, _ = evaluar_lstm_en_folds(
            df, inner_folds, configuracion, params, trial=trial
        )
        epochs_by_trial[trial.number] = best_epochs
        return score

    study.optimize(objective, n_trials=LSTM_TRIALS, show_progress_bar=False)
    winning_epochs = epochs_by_trial[study.best_trial.number]
    chosen_epochs = max(1, int(round(median(winning_epochs))))
    path = OUTPUT_DIR / (
        f"hibrido_lstm_nested_{slug_configuracion(configuracion)}_fold{outer_fold}.csv"
    )
    study.trials_dataframe().to_csv(path, index=False, encoding="utf-8-sig")
    print(
        f"  LSTM base seleccionada solo dentro del fold externo {outer_fold} | "
        f"sMAPE interno={study.best_value:.4f} | épocas={chosen_epochs} | "
        f"parámetros={study.best_params}", flush=True,
    )
    return dict(study.best_params), chosen_epochs


def preparar_cache_hibrido(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    configuracion: str,
    lstm_params: dict,
    lstm_epochs: int,
) -> list[dict]:
    cache: list[dict] = []
    imprimir_subtitulo(f"LSTM-XGBOOST - {etiqueta_configuracion(configuracion)} | PREPARACIÓN OOF")

    for fold_number, (outer_train, outer_val) in enumerate(folds, start=1):
        print(
            f"Preparando fold externo {fold_number}/{len(folds)} | "
            f"train={len(outer_train)} | validación={len(outer_val)}",
            flush=True,
        )
        if HYBRID_NESTED_LSTM_SELECTION:
            local_params, local_epochs = seleccionar_lstm_solo_con_entrenamiento_externo(
                df, outer_train, configuracion, fold_number
            )
        else:
            local_params, local_epochs = lstm_params, lstm_epochs
        oof_series = generar_oof_lstm(
            df,
            outer_train,
            configuracion,
            local_params,
            local_epochs,
        )
        oof_rows = oof_series.index.to_numpy(dtype=int)
        oof_pred = oof_series.to_numpy(dtype=float)
        oof_true = df.loc[oof_rows, TARGET].to_numpy(dtype=float)
        X_res_train, feature_names = construir_features_residuales(
            df,
            oof_rows,
            oof_pred,
            configuracion,
        )
        if not np.isfinite(oof_true).all():
            raise AssertionError("El corrector OOF recibió etiquetas imputadas.")
        if np.any(oof_rows >= outer_val[0]):
            raise AssertionError("El corrector entrenaría con validación externa o posterior.")
        y_res_train = oof_true - oof_pred

        base_epochs = int(local_epochs)
        base_pred_val, y_val, val_rows = entrenar_lstm_y_predecir_bloque(
            df,
            outer_train,
            outer_val,
            configuracion,
            local_params,
            base_epochs,
        )
        X_res_val, _ = construir_features_residuales(
            df,
            val_rows,
            base_pred_val,
            configuracion,
        )

        print(
            f"  Fold externo {fold_number}/{len(folds)} preparado | "
            f"residuos OOF={len(y_res_train)} | validación={len(y_val)} | "
            f"épocas LSTM base={base_epochs}",
            flush=True,
        )
        cache.append(
            {
                "fold": fold_number,
                "X_res_train": X_res_train,
                "y_res_train": y_res_train,
                "X_res_val": X_res_val,
                "base_pred_val": base_pred_val,
                "y_val": y_val,
                "val_rows": val_rows,
                "feature_names": feature_names,
                "base_epochs": base_epochs,
                "lstm_params_fold": local_params,
                "nested_lstm": HYBRID_NESTED_LSTM_SELECTION,
            }
        )

    return cache



def seleccionar_hibrido(
    cache: list[dict],
    configuracion: str,
) -> tuple[dict, optuna.Study, pd.DataFrame, dict]:
    fold_records: dict[int, list[dict]] = {}
    trial_histories: dict[int, list[dict]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = sugerir_parametros_xgboost(trial)
        scores: list[float] = []
        records: list[dict] = []
        histories: list[dict] = []
        print(
            f"Trial {trial.number + 1}/{HYBRID_XGB_TRIALS} | {formatear_parametros(params)}",
            flush=True,
        )

        for fold_data in cache:
            y_res_val = fold_data["y_val"] - fold_data["base_pred_val"]
            model = crear_xgboost(params)
            model.fit(
                fold_data["X_res_train"],
                fold_data["y_res_train"],
                eval_set=[
                    (fold_data["X_res_train"], fold_data["y_res_train"]),
                    (fold_data["X_res_val"], y_res_val),
                ],
                verbose=False,
            )
            correction = model.predict(fold_data["X_res_val"])
            hybrid_pred = fold_data["base_pred_val"] + correction
            metrics = calcular_metricas(
                fold_data["y_val"],
                hybrid_pred,
            )
            scores.append(metrics["smape"])

            evals = model.evals_result()
            histories.append(
                {
                    "fold": int(fold_data["fold"]),
                    "train_rmse": [float(v) for v in evals["validation_0"]["rmse"]],
                    "val_rmse": [float(v) for v in evals["validation_1"]["rmse"]],
                }
            )

            print(
                f"  Fold {fold_data['fold']}/{len(cache)} | "
                f"sMAPE={metrics['smape']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}",
                flush=True,
            )
            records.append(
                {
                    "modelo": "LSTM-XGBoost",
                    "configuracion": etiqueta_configuracion(configuracion),
                    "trial": trial.number,
                    "fold": fold_data["fold"],
                    **metrics,
                    "n": len(fold_data["y_val"]),
                    "epocas_lstm_base": fold_data["base_epochs"],
                }
            )
            trial.report(float(np.mean(scores)), step=fold_data["fold"])
            if trial.should_prune():
                raise optuna.TrialPruned()

        fold_records[trial.number] = records
        trial_histories[trial.number] = histories
        trial_mean = float(np.mean(scores))
        print(f"  sMAPE promedio del trial: {trial_mean:.4f}", flush=True)
        return trial_mean

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"lstm_xgboost_{slug_configuracion(configuracion)}",
    )
    imprimir_subtitulo(
        f"LSTM-XGBOOST - {etiqueta_configuracion(configuracion)} | XGBOOST RESIDUAL"
    )
    study.optimize(
        objective,
        n_trials=HYBRID_XGB_TRIALS,
        show_progress_bar=False,
    )
    print(
        f"Mejor LSTM-XGBoost-{etiqueta_configuracion(configuracion)}: sMAPE CV={study.best_value:.4f} | "
        f"{formatear_parametros(study.best_params)}",
        flush=True,
    )

    best_trial = study.best_trial.number
    if best_trial not in fold_records or best_trial not in trial_histories:
        raise RuntimeError(
            f"No se conservaron resultados completos del mejor trial híbrido-{etiqueta_configuracion(configuracion)}."
        )

    return (
        dict(study.best_params),
        study,
        pd.DataFrame(fold_records[best_trial]),
        trial_histories[best_trial][-1],
    )

def entrenar_hibrido_final(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    configuracion: str,
    lstm_params: dict,
    lstm_epochs: int,
    residual_params: dict,
) -> dict:
    print(f"Reajustando LSTM-XGBoost-{etiqueta_configuracion(configuracion)} con todo el conjunto de implementación (80 %)...", flush=True)
    oof_series = generar_oof_lstm(
        df,
        train_indices,
        configuracion,
        lstm_params,
        lstm_epochs,
    )
    rows = oof_series.index.to_numpy(dtype=int)
    base_pred = oof_series.to_numpy(dtype=float)
    y_true = df.loc[rows, TARGET].to_numpy(dtype=float)
    X_res, feature_names = construir_features_residuales(
        df,
        rows,
        base_pred,
        configuracion,
    )
    if not np.isfinite(y_true).all():
        raise AssertionError("El reajuste híbrido recibió objetivos no observados.")
    residual_target = y_true - base_pred

    residual_model = crear_xgboost(residual_params)
    residual_model.fit(X_res, residual_target)

    model_path = MODEL_DIR / f"lstm_xgboost_{slug_configuracion(configuracion)}_residual.joblib"
    joblib.dump(residual_model, model_path)
    print(f"LSTM-XGBoost-{etiqueta_configuracion(configuracion)} reajustado y guardado.", flush=True)

    return {
        "corrector_residual": str(model_path),
        "modelo_base_lstm": str(MODEL_DIR / f"lstm_{slug_configuracion(configuracion)}.pt"),
        "escaladores_base_lstm": str(
            MODEL_DIR / f"lstm_{slug_configuracion(configuracion)}_scalers.joblib"
        ),
        "columnas_corrector": feature_names,
        "n_residuos_oof": int(len(rows)),
        "epocas_oof": int(lstm_epochs),
    }



def guardar_historial_red_experimentacion(
    history: dict,
    nombre_modelo: str,
    configuracion: str,
) -> Path:
    """
    Guarda la curva del fold 5 del mejor trial para LSTM o TCN.
    Se hace para las dos configuraciones: Histórico e Histórico + abastecimiento.
    """
    if nombre_modelo not in {"LSTM", "TCN"}:
        raise ValueError("Solo se generan curvas de pérdida para LSTM y TCN.")

    slug = slug_configuracion(configuracion)
    path = HISTORY_DIR / f"{nombre_modelo.lower()}_curva_{slug}.csv"

    pd.DataFrame(
        {
            "epoca": np.arange(1, len(history["train_loss"]) + 1),
            "entrenamiento": history["train_loss"],
            "validacion": history["val_loss"],
            "epoca_optima": int(history["best_epoch"]),
            "fold": int(history.get("fold", 5)),
            "configuracion": etiqueta_configuracion(configuracion),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")

    generar_figura_curva_red_individual(
        archivo_historial=path,
        nombre_modelo=nombre_modelo,
        configuracion=configuracion,
    )
    return path


def generar_figura_curva_red_individual(
    archivo_historial: Path,
    nombre_modelo: str,
    configuracion: str,
) -> Path:
    """
    Figura individual de pérdida de entrenamiento y validación.
    Se usa únicamente para LSTM y TCN, que son los modelos de Deep Learning
    entrenados por épocas en este estudio.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(archivo_historial)
    fold = int(data["fold"].iloc[0]) if "fold" in data.columns else 5
    best_epoch = int(data["epoca_optima"].iloc[0])

    panel_code = {
        ("LSTM", CONFIG_HISTORICO): "4a",
        ("LSTM", CONFIG_HISTORICO_ABASTECIMIENTO): "4b",
        ("TCN", CONFIG_HISTORICO): "4c",
        ("TCN", CONFIG_HISTORICO_ABASTECIMIENTO): "4d",
    }.get((nombre_modelo, configuracion), f"4_{nombre_modelo.lower()}_{slug_configuracion(configuracion)}")

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(data["epoca"], data["entrenamiento"], label="Entrenamiento")
    ax.plot(data["epoca"], data["validacion"], label="Validación")
    ax.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.1,
        label=f"Época óptima ({best_epoch})",
    )
    ax.set_title(
        f"{nombre_modelo}: pérdida de entrenamiento y validación\n"
        f"{etiqueta_configuracion(configuracion)}, fold {fold}"
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE del objetivo escalado")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    path = FIGURE_DIR / (
        f"Figura_{panel_code}_{nombre_modelo}_"
        f"{slug_configuracion(configuracion)}_curva_perdida.png"
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura individual de Deep Learning generada: {path}", flush=True)
    return path


def generar_figura_curvas_deep_learning() -> Path:
    """Cuatro curvas del fold 5 con exactamente las mismas escalas de ejes.

    Se muestran los mejores ensayos por configuración, no se selecciona
    retrospectivamente el fold más favorable. Otros folds quedan auditados.
    """
    archivos = [
        ("LSTM", CONFIG_HISTORICO, HISTORY_DIR / "lstm_curva_historico.csv"),
        ("LSTM", CONFIG_HISTORICO_ABASTECIMIENTO,
         HISTORY_DIR / "lstm_curva_historico_abastecimiento.csv"),
        ("TCN", CONFIG_HISTORICO, HISTORY_DIR / "tcn_curva_historico.csv"),
        ("TCN", CONFIG_HISTORICO_ABASTECIMIENTO,
         HISTORY_DIR / "tcn_curva_historico_abastecimiento.csv"),
    ]
    datos = []
    for modelo, config, ruta in archivos:
        if not ruta.exists():
            raise FileNotFoundError(f"Falta el historial de {modelo} - {config}: {ruta}")
        d = pd.read_csv(ruta)
        if d.empty or not {"epoca", "entrenamiento", "validacion", "epoca_optima", "fold"}.issubset(d):
            raise ValueError(f"Historial incompleto: {ruta}")
        datos.append(d)
    max_epoch = max(int(d.epoca.max()) for d in datos)
    perdidas = np.concatenate([d[c].to_numpy(dtype=float)
                               for d in datos for c in ("entrenamiento", "validacion")])
    positivas = perdidas[np.isfinite(perdidas) & (perdidas > 0)]
    if len(positivas) != len(perdidas):
        raise ValueError("La pérdida MSE debe ser positiva y finita para la escala logarítmica.")
    lo = 10 ** np.floor(np.log10(float(positivas.min())))
    hi = 10 ** np.ceil(np.log10(float(positivas.max())))
    if lo == hi:
        hi *= 10.0
    from matplotlib.ticker import MultipleLocator
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.9), sharex=True, sharey=True)
    for ax, panel, (modelo, config, _), d in zip(
        axes.flat, ("(a)", "(b)", "(c)", "(d)"), archivos, datos
    ):
        mejor = int(d.epoca_optima.iloc[0])
        fold = int(d.fold.iloc[0])
        ax.plot(d.epoca, d.entrenamiento, color="#23639a", lw=1.5,
                label="Entrenamiento")
        ax.plot(d.epoca, d.validacion, color="#c16f2c", lw=1.5,
                label="Validación")
        ax.axvline(mejor, linestyle="--", color="#424242", lw=1.1,
                   label=f"Mejor época: {mejor}")
        ax.set_title(f"{panel} {modelo} - {etiqueta_configuracion(config)} (fold {fold})", fontsize=10)
        ax.set_xlim(0, max_epoch + 2)
        ax.set_ylim(lo, hi)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.grid(alpha=0.15, which="major")
        ax.legend(fontsize=8, loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("Época")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE del objetivo escalado (escala log)")
    fig.suptitle("Pérdida de entrenamiento y validación de LSTM y TCN - fold 5", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "Figura_4_curvas_perdida_deep_learning.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generar_resumen_epocas_folds()
    print(f"Figura comparable del fold 5 generada: {path}", flush=True)
    return path


def generar_resumen_epocas_folds() -> Path:
    """Tabla y figura opcional de los cinco folds del mejor ensayo temporal."""
    filas = []
    for modelo in ("LSTM", "TCN"):
        for config in CONFIGURACIONES:
            ruta = HISTORY_DIR / f"{modelo.lower()}_curvas_{slug_configuracion(config)}.csv"
            if not ruta.exists():
                raise FileNotFoundError(f"Falta el historial de los cinco folds: {ruta}")
            d = pd.read_csv(ruta)
            required = {"fold", "epoca", "epoca_optima", "train_loss", "val_loss"}
            if not required.issubset(d):
                raise ValueError(f"Columnas de historial incompletas en {ruta}")
            for fold, g in d.groupby("fold"):
                opt = int(g.epoca_optima.iloc[0])
                if not g.epoca_optima.eq(opt).all():
                    raise ValueError("Las épocas óptimas deben ser constantes por fold")
                filas.append({"modelo": modelo, "configuracion": etiqueta_configuracion(config),
                              "fold": int(fold), "mejor_epoca": opt,
                              "epocas_ejecutadas": int(g.epoca.max()),
                              "mse_validacion_optimo": float(g.loc[g.epoca.eq(opt), "val_loss"].iloc[0])})
    resumen = pd.DataFrame(filas).sort_values(["modelo", "configuracion", "fold"])
    if len(resumen) != 20:
        raise ValueError("Se esperaban 4 configuraciones y 5 folds por configuración")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    destino_csv = TABLE_DIR / "Tabla_articulo_epocas_optimas_5_folds.csv"
    resumen.to_csv(destino_csv, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    colores = ("#23639a", "#4a8cbe", "#b97327", "#d3a15d")
    for k, ((modelo, config), sub) in enumerate(resumen.groupby(["modelo", "configuracion"], sort=True)):
        ax.scatter(sub.fold.to_numpy() + offsets[k], sub.mejor_epoca.to_numpy(),
                   s=48, color=colores[k], label=f"{modelo} - {config}")
    ax.set_xticks(range(1, 6))
    ax.set(xlabel="Fold temporal", ylabel="Mejor época según MSE de validación",
           title="Épocas óptimas en los cinco folds del mejor ensayo")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIGURE_DIR / "Figura_articulo_epocas_por_fold_opcional.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Resumen de épocas por fold: {destino_csv}", flush=True)
    return path


def generar_figura_folds(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    test_indices: np.ndarray,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.6))

    color_train = "C0"
    color_val = "C1"
    color_test = "0.85"

    for pos, (train_idx, val_idx) in enumerate(folds, start=1):
        train_start = mdates.date2num(df.loc[train_idx[0], DATE])
        train_end = mdates.date2num(df.loc[train_idx[-1], DATE])
        val_start = mdates.date2num(df.loc[val_idx[0], DATE])
        val_end = mdates.date2num(df.loc[val_idx[-1], DATE])
        ax.barh(
            pos,
            train_end - train_start + 1,
            left=train_start,
            color=color_train,
            label="Entrenamiento" if pos == 1 else None,
        )
        ax.barh(
            pos,
            val_end - val_start + 1,
            left=val_start,
            color=color_val,
            label="Validación" if pos == 1 else None,
        )

    test_start = mdates.date2num(df.loc[test_indices[0], DATE])
    test_end = mdates.date2num(df.loc[test_indices[-1], DATE])
    ax.axvspan(
        test_start,
        test_end + 1,
        facecolor=color_test,
        alpha=0.7,
        hatch="//",
        edgecolor="0.45",
        label="Conjunto final de prueba reservado",
    )
    ax.axvline(test_start, linestyle="--", linewidth=1.2, color="0.35")

    ax.set_yticks(
        range(1, len(folds) + 1),
        [f"Fold {i}" for i in range(1, len(folds) + 1)],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Fold temporal")
    ax.set_title("Ejecución real de la validación temporal expansiva en los cinco folds")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = FIGURE_DIR / "Figura_1_folds_temporales.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura de experimentación generada: {path}", flush=True)
    return path


def generar_figura_serie_diaria_completa(
    df: pd.DataFrame,
    implementation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Path:
    """
    Muestra las dos series diarias que alimentan el experimento después del
    preprocesamiento de la Fase 2: precio y abastecimiento. También deja visible
    el corte cronológico entre implementación (80 %) y prueba final (20 %).
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fechas = df[DATE]
    test_start = df.loc[int(test_indices[0]), DATE]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    axes[0].plot(
        fechas,
        df[PRICE_ORIGINAL],
        linewidth=0.9,
        label="Precio promedio mayorista observado",
    )
    axes[0].axvline(
        test_start,
        linestyle="--",
        linewidth=1.1,
        label="Inicio de prueba final (20 %)",
    )
    axes[0].set_ylabel("Precio (PEN/kg)")
    axes[0].set_title(
        "Series diarias utilizadas en el experimento"
    )
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.15)

    axes[1].plot(
        fechas,
        df[VOLUME_ORIGINAL],
        linewidth=0.9,
        label="Abastecimiento observado",
    )
    axes[1].axvline(
        test_start,
        linestyle="--",
        linewidth=1.1,
        label="Inicio de prueba final (20 %)",
    )
    axes[1].set_ylabel("Abastecimiento (t)")
    axes[1].set_xlabel("Fecha")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.15)

    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = FIGURE_DIR / "Figura_2_series_diarias_completas.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura de series diarias generada: {path}", flush=True)
    return path


def generar_figura_ventana_real(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    window: int,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    end_idx = int(train_indices[-2])
    start_idx = end_idx - int(window) + 1
    if start_idx < int(train_indices[0]):
        raise RuntimeError(
            "No hay observaciones suficientes para construir la figura de ventana temporal."
        )

    rows = np.arange(start_idx, end_idx + 1, dtype=int)
    dates = df.loc[rows, DATE]
    origin_date = df.loc[end_idx, DATE]
    target_date = df.loc[end_idx, TARGET_DATE]
    target_value = float(df.loc[end_idx, TARGET])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    axes[0].plot(
        dates,
        df.loc[rows, PRICE],
        marker="o",
        markersize=2.5,
        label="Precio disponible hasta t",
    )
    axes[0].scatter(
        [target_date],
        [target_value],
        marker="X",
        s=75,
        label="Precio objetivo t+1",
        zorder=5,
    )
    axes[0].axvline(target_date, linestyle="--", linewidth=1.0, color="0.35")
    axes[0].set_ylabel("Precio (PEN/kg)")
    axes[0].legend(loc="best")
    axes[0].set_title(
        "Ejemplo real de construcción de una ventana temporal secuencial para el pronóstico en t+1\n"
        f"TCN seleccionado | ventana={window} días | Histórico + abastecimiento"
    )

    axes[1].plot(
        dates,
        df.loc[rows, VOLUME],
        marker="o",
        markersize=2.5,
        label="Abastecimiento disponible hasta t",
    )
    axes[1].axvline(target_date, linestyle="--", linewidth=1.0, color="0.35")
    axes[1].set_ylabel("Abastecimiento (t)")
    axes[1].set_xlabel("Fecha")
    axes[1].legend(loc="best")
    axes[1].text(
        0.01,
        0.03,
        "La entrada también incluye las variables calendario cíclicas de cada fecha histórica.",
        transform=axes[1].transAxes,
        fontsize=9,
        va="bottom",
    )

    axes[0].annotate(
        "t",
        xy=(origin_date, float(df.loc[end_idx, PRICE])),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    axes[0].annotate(
        "t+1",
        xy=(target_date, target_value),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    path = FIGURE_DIR / "Figura_3_ventana_temporal_t1.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura de experimentación generada: {path}", flush=True)
    return path


def guardar_resultados_optuna(
    study: optuna.Study,
    path_trials: Path,
    best_fold_results: pd.DataFrame,
    path_folds: Path,
) -> None:
    study.trials_dataframe().to_csv(
        path_trials,
        index=False,
        encoding="utf-8-sig",
    )
    best_fold_results.to_csv(
        path_folds,
        index=False,
        encoding="utf-8-sig",
    )
    trials = study.trials_dataframe()
    if "value" in trials and "number" in trials:
        completed = trials.loc[np.isfinite(pd.to_numeric(trials["value"], errors="coerce"))].copy()
        if not completed.empty:
            completed["mejor_hasta_trial"] = completed["value"].cummin()
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.plot(completed["number"] + 1, completed["value"], "o", ms=4, alpha=.5, label="Trial completado")
            ax.plot(completed["number"] + 1, completed["mejor_hasta_trial"], linewidth=2, label="Mejor sMAPE acumulado")
            ax.set(title=f"Optuna: {study.study_name}", xlabel="Número de trial", ylabel="sMAPE de validación (%)")
            ax.grid(alpha=.18); ax.legend(); fig.tight_layout()
            fig.savefig(FIGURE_DIR / f"Figura_optuna_{study.study_name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def construir_resumen_validacion(
    result_tables: list[pd.DataFrame],
    baseline_table: pd.DataFrame,
) -> pd.DataFrame:
    tables = [baseline_table] + result_tables
    rows: list[dict] = []

    for table in tables:
        if table.empty:
            continue
        for (model, config), group in table.groupby(
            ["modelo", "configuracion"],
            dropna=False,
        ):
            rows.append(
                {
                    "modelo": model,
                    "configuracion": config,
                    "smape_promedio": float(group["smape"].mean()),
                    "smape_desviacion": float(group["smape"].std(ddof=0)),
                    "mae_promedio": float(group["mae"].mean()),
                    "rmse_promedio": float(group["rmse"].mean()),
                    "folds": int(group["fold"].nunique()),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["modelo", "configuracion"]
    ).reset_index(drop=True)


def guardar_tablas_y_figura_resumen_fase3(summary: pd.DataFrame, configs: dict,
                                         df: pd.DataFrame) -> None:
    """Sólo CV, sin evaluar el 20 % reservado."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "Tabla_3_resultados_validacion_5_folds.csv", index=False, encoding="utf-8-sig")
    params_rows=[]
    for name, data in configs.items():
        cfg = data if isinstance(data, dict) else {}
        params_rows.append({"modelo_configuracion":name, "hiperparametros_y_seleccion":json.dumps(convertir_json(cfg),ensure_ascii=False)})
    pd.DataFrame(params_rows).to_csv(TABLE_DIR / "Tabla_3b_hiperparametros_seleccionados.csv",index=False,encoding="utf-8-sig")
    if summary.empty: return
    fig,ax=plt.subplots(figsize=(11,5.8))
    tab=summary.copy()
    tab["clave"] = tab["modelo"]+" - "+tab["configuracion"]
    tab=tab.sort_values("smape_promedio", ascending=False)
    ax.barh(tab["clave"],tab["smape_promedio"],xerr=tab["smape_desviacion"],capsize=2)
    ax.set_xlabel("sMAPE medio de validación (%) ± desviación entre folds")
    ax.set_title("Resultados de validación temporal (no prueba final)")
    ax.grid(axis="x",alpha=.17);fig.tight_layout()
    fig.savefig(FIGURE_DIR/"Figura_5_resumen_smape_validacion.png",dpi=300,bbox_inches="tight");plt.close(fig)


def main() -> None:
    inicio = perf_counter()
    fijar_semilla(SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


    input_digest = hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest() if INPUT_FILE.exists() else None
    guardar_json(OUTPUT_DIR / "fase3_estado.json", {
        "estado": "EN_CURSO",
        "inicio": datetime.now().isoformat(timespec="seconds"),
        "protocolo_preprocesamiento": PREPROCESSING_VERSION,
        "sha256_dataset_entrada": input_digest,
        "advertencia": "No utilizar modelos/metricas anteriores hasta finalizar el reajuste causal.",
    })

    imprimir_titulo("FASE 3 - IMPLEMENTACIÓN: ENTRENAMIENTO Y VALIDACIÓN TEMPORAL DE LOS MODELOS")
    print(f"Dispositivo PyTorch: {DEVICE}", flush=True)
    print(f"Archivo de entrada: {INPUT_FILE}", flush=True)

    print("\nCargando salida de la Fase 2...", flush=True)
    df = cargar_dataset()
    describir_variables_por_modelo(df)
    print(f"Dataset cargado: {len(df)} observaciones.", flush=True)
    implementation_indices, test_indices, partition_metadata = separar_implementacion_prueba(df)
    print(
        f"Partición cronológica: implementación={len(implementation_indices)} ({IMPLEMENTATION_RATIO:.0%}) | "
        f"conjunto final de prueba={len(test_indices)} ({1.0 - IMPLEMENTATION_RATIO:.0%}).",
        flush=True,
    )
    print(
        f"Implementación: {partition_metadata['origen_implementacion_inicio']} a "
        f"{partition_metadata['origen_implementacion_fin']}", flush=True
    )
    print(
        f"Conjunto final de prueba: {partition_metadata['origen_prueba_inicio']} a "
        f"{partition_metadata['origen_prueba_fin']}", flush=True
    )

    folds = construir_folds_temporales(implementation_indices)
    print(f"Validación temporal interna: {len(folds)} folds expansivos.", flush=True)
    for fold_number, (fold_train, fold_val) in enumerate(folds, start=1):
        print(
            f"  Fold {fold_number}: entrenamiento={len(fold_train)} | validación={len(fold_val)} | "
            f"entrenamiento fin={df.loc[fold_train[-1], DATE].date()} | "
            f"validación={df.loc[fold_val[0], DATE].date()} a {df.loc[fold_val[-1], DATE].date()}",
            flush=True,
        )
    guardar_json(OUTPUT_DIR / "particion_temporal.json", partition_metadata)
    guardar_folds(df, folds)
    guardar_auditoria_temporal(df, implementation_indices, test_indices, folds)
    comprobar_muestras_comunes(df, folds)
    print(
        f"Objetivos observados: desarrollo={partition_metadata['n_objetivos_observados_desarrollo']}, "
        f"prueba={partition_metadata['n_objetivos_observados_prueba']} | "
        f"sin precio real: desarrollo={partition_metadata['n_objetivos_ausentes_desarrollo']}, "
        f"prueba={partition_metadata['n_objetivos_ausentes_prueba']}", flush=True,
    )
    generar_figura_folds(df, folds, test_indices)
    generar_figura_serie_diaria_completa(df, implementation_indices, test_indices)

    baselines = evaluar_baselines(df, folds)
    best_configs: dict[str, dict] = {}
    model_artifacts: dict[str, dict] = {}
    validation_tables: list[pd.DataFrame] = []

    for configuracion in CONFIGURACIONES:
        etiqueta = etiqueta_configuracion(configuracion)
        slug = slug_configuracion(configuracion)
        imprimir_titulo(f"CONFIGURACIÓN: {etiqueta}")

        with registrar_bloque(f"SARIMAX | validación | {etiqueta}"):
            best_sarimax, sarimax_prescreen, sarimax_folds = seleccionar_sarimax(df, folds, configuracion)
        sarimax_prescreen.to_csv(OUTPUT_DIR / f"sarimax_preseleccion_{slug}.csv", index=False, encoding="utf-8-sig")
        sarimax_folds.to_csv(OUTPUT_DIR / f"sarimax_validacion_{slug}.csv", index=False, encoding="utf-8-sig")
        best_configs[clave_modelo("SARIMAX", configuracion)] = best_sarimax
        with registrar_bloque(f"SARIMAX | reajuste con implementación completa (80 %) | {etiqueta}"):
            model_artifacts[clave_modelo("SARIMAX", configuracion)] = entrenar_sarimax_final(df, implementation_indices, configuracion, best_sarimax)
        validation_tables.append(sarimax_folds[sarimax_folds["candidato"] == int(best_sarimax["candidato"])].copy())

        with registrar_bloque(f"XGBoost | validación | {etiqueta}"):
            best_xgb, xgb_study, xgb_folds, xgb_history = seleccionar_xgboost(df, folds, configuracion)
        guardar_resultados_optuna(xgb_study, OUTPUT_DIR / f"xgboost_busqueda_{slug}.csv", xgb_folds, OUTPUT_DIR / f"xgboost_validacion_{slug}.csv")
        best_configs[clave_modelo("XGBoost", configuracion)] = {**best_xgb, "smape_validacion": float(xgb_study.best_value)}
        with registrar_bloque(f"XGBoost | reajuste con implementación completa (80 %) | {etiqueta}"):
            model_artifacts[clave_modelo("XGBoost", configuracion)] = entrenar_xgboost_final(df, implementation_indices, configuracion, best_xgb)
        validation_tables.append(xgb_folds)

        with registrar_bloque(f"LSTM | validación | {etiqueta}"):
            best_lstm, lstm_epochs, lstm_study, lstm_folds, lstm_history = seleccionar_lstm(df, folds, configuracion)
        guardar_resultados_optuna(lstm_study, OUTPUT_DIR / f"lstm_busqueda_{slug}.csv", lstm_folds, OUTPUT_DIR / f"lstm_validacion_{slug}.csv")
        best_configs[clave_modelo("LSTM", configuracion)] = {**best_lstm, "learning_rate": LSTM_LEARNING_RATE, "epocas_reajuste": lstm_epochs, "smape_validacion": float(lstm_study.best_value)}
        guardar_historial_red_experimentacion(
            lstm_history,
            "LSTM",
            configuracion,
        )
        with registrar_bloque(f"LSTM | reajuste con implementación completa (80 %) | {etiqueta}"):
            model_artifacts[clave_modelo("LSTM", configuracion)] = entrenar_lstm_final(df, implementation_indices, configuracion, best_lstm, lstm_epochs)
        validation_tables.append(lstm_folds)

        with registrar_bloque(f"TCN | validación | {etiqueta}"):
            best_tcn, tcn_epochs, tcn_study, tcn_folds, tcn_history = seleccionar_tcn(df, folds, configuracion)
        guardar_resultados_optuna(tcn_study, OUTPUT_DIR / f"tcn_busqueda_{slug}.csv", tcn_folds, OUTPUT_DIR / f"tcn_validacion_{slug}.csv")
        guardar_historial_red_experimentacion(
            tcn_history,
            "TCN",
            configuracion,
        )
        if configuracion == CONFIG_HISTORICO_ABASTECIMIENTO:
            generar_figura_ventana_real(
                df,
                implementation_indices,
                int(best_tcn["window"]),
            )
        epocas_optimas_folds=[int(v) for v in tcn_folds["epoca_optima"].tolist()]
        best_configs[clave_modelo("TCN", configuracion)] = {
            **best_tcn, "n_capas_convolucionales":2*int(best_tcn["n_blocks"]),
            "dilataciones":[2**i for i in range(int(best_tcn["n_blocks"]))],
            "learning_rate":TCN_LEARNING_RATE,"batch_size":TCN_BATCH_SIZE,"max_epochs":TCN_MAX_EPOCHS,"patience":TCN_PATIENCE,
            "variables_calendario":True,"arquitectura_tcn":TCN_ARCHITECTURE_ID,"epocas_optimas_folds":epocas_optimas_folds,
            "epocas_reajuste":int(tcn_epochs),"regla_epocas_reajuste":"mediana de best_epoch de los 5 folds del mejor trial",
            "campo_receptivo":calcular_campo_receptivo_tcn(int(best_tcn["n_blocks"]),int(best_tcn["kernel_size"])),
            "smape_validacion":float(tcn_study.best_value),
        }
        with registrar_bloque(f"TCN | reajuste con implementación completa (80 %) | {etiqueta}"):
            model_artifacts[clave_modelo("TCN", configuracion)] = entrenar_tcn_final(df,implementation_indices,configuracion,best_tcn,tcn_epochs,smape_cv=float(tcn_study.best_value))
        validation_tables.append(tcn_folds)

        with registrar_bloque(f"LSTM-XGBoost | preparación y validación | {etiqueta}"):
            hybrid_cache = preparar_cache_hibrido(df, folds, configuracion, best_lstm, lstm_epochs)
            best_residual, hybrid_study, hybrid_folds, hybrid_history = seleccionar_hibrido(hybrid_cache, configuracion)
        guardar_resultados_optuna(hybrid_study, OUTPUT_DIR / f"hibrido_busqueda_{slug}.csv", hybrid_folds, OUTPUT_DIR / f"hibrido_validacion_{slug}.csv")
        best_configs[clave_modelo("LSTM-XGBoost", configuracion)] = {
            "lstm": best_lstm,
            "xgboost_residual": best_residual,
            "smape_validacion": float(hybrid_study.best_value),
            "seleccion_anidada_lstm_por_fold": HYBRID_NESTED_LSTM_SELECTION,
        }
        with registrar_bloque(f"LSTM-XGBoost | reajuste con implementación completa (80 %) | {etiqueta}"):
            model_artifacts[clave_modelo("LSTM-XGBoost", configuracion)] = entrenar_hibrido_final(
                df, implementation_indices, configuracion, best_lstm, lstm_epochs, best_residual
            )
        validation_tables.append(hybrid_folds)
        base_fold = lstm_folds[["fold","smape"]].rename(columns={"smape":"smape_LSTM_base"})
        comparacion_hibrido=hybrid_folds[["fold","smape"]].rename(columns={"smape":"smape_hibrido"}).merge(base_fold,on="fold",validate="one_to_one")
        comparacion_hibrido["delta_hibrido_menos_lstm"] = comparacion_hibrido["smape_hibrido"]-comparacion_hibrido["smape_LSTM_base"]
        comparacion_hibrido.to_csv(OUTPUT_DIR/f"diagnostico_hibrido_folds_{slug}.csv",index=False,encoding="utf-8-sig")
        if EJECUTAR_DIAGNOSTICOS_AMPLIADOS and configuracion == CONFIG_HISTORICO_ABASTECIMIENTO:
            with registrar_bloque("Sensibilidad exploratoria - alpha y bloque 21 días"):
                sensibilidad_alpha_en_pronostico(df,folds,best_xgb)
                validar_sensibilidad_21_dias(df,folds,best_xgb)

    generar_figura_curvas_deep_learning()

    guardar_json(OUTPUT_DIR / "mejores_configuraciones.json", best_configs)
    guardar_json(OUTPUT_DIR / "artefactos_modelos.json", normalizar_rutas_artefactos(model_artifacts))

    summary = construir_resumen_validacion(validation_tables, baselines)
    summary.to_csv(OUTPUT_DIR / "resumen_validacion.csv", index=False, encoding="utf-8-sig")
    guardar_tablas_y_figura_resumen_fase3(summary, best_configs, df)

    elapsed = perf_counter() - inicio
    imprimir_titulo("RESUMEN FINAL DE IMPLEMENTACIÓN")
    if not summary.empty:
        print(summary.to_string(index=False), flush=True)
    print(f"\nFase 3 completada en {elapsed / 60.0:.1f} minutos.", flush=True)
    print(f"Implementación: {len(implementation_indices)} observaciones destinadas a entrenamiento y validación.", flush=True)
    print(f"Conjunto final de prueba reservado: {len(test_indices)} observaciones.", flush=True)
    print(f"Resultados de validación: {OUTPUT_DIR}", flush=True)
    print(f"Figuras de Experimentación: {FIGURE_DIR}", flush=True)
    # Ablación exploratoria por grupos: reentrena con exactamente el mismo protocolo.
    # La función registra sus resultados sin sobrescribir los modelos principales.
    from fase3b_ablacion import ejecutar_ablacion as ejecutar_ablacion_fase3
    with registrar_bloque("ABLACIÓN - Calendario y Calendario + abastecimiento"):
        ejecutar_ablacion_fase3()

    guardar_json(OUTPUT_DIR / "fase3_estado.json", {
        "estado": "COMPLETADA",
        "fin": datetime.now().isoformat(timespec="seconds"),
        "protocolo_preprocesamiento": PREPROCESSING_VERSION,
        "sha256_dataset_entrada": input_digest,
        "sha256_protocolo_fase2":hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest(),
        "alpha_ewma":json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))["alpha_ewma"],
        "regla_seleccion_alpha":"minimo_MAE_promedio_equilibrado_por_duracion",
        "version_python":platform.python_version(),
        "versiones_ejecucion": {
            "torch":torch.__version__,"cuda_disponible":torch.cuda.is_available(),
            "cuda_version":torch.version.cuda,"optuna":optuna.__version__,
            "xgboost":__import__("xgboost").__version__,
            "pandas":pd.__version__,"numpy":np.__version__,
            "statsmodels":__import__("statsmodels").__version__,
        },
        "diagnosticos_ampliados":EJECUTAR_DIAGNOSTICOS_AMPLIADOS,
        "folds": N_SPLITS,
        "proporcion_desarrollo": IMPLEMENTATION_RATIO,
        "lstm_hibrido_seleccion_anidada": HYBRID_NESTED_LSTM_SELECTION,
        "resultados": str(OUTPUT_DIR),
        "ablacion_grupos": ["Calendario", "Calendario + abastecimiento"],
    })
    print("El conjunto final de prueba no fue evaluado en esta fase.", flush=True)
    print("El instante real de publicación de SISAP debe verificarse antes de afirmar ausencia total de fuga temporal.", flush=True)


if __name__ == "__main__":
    ejecutar_con_log(main, "FASE 3", LOG_FILE)
