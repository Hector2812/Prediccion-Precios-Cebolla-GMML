# SISAP - Series temporales: http://sistemas.midagri.gob.pe/sisap/portal/index.php

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime
import sys
import traceback
import json
import hashlib
import re

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "Fase_1_log.txt"
DOWNLOAD_DIR = ROOT / "data descargada"
OUTPUT_DIR = ROOT / "data consolidada"

URL_SISAP = "http://sistemas.midagri.gob.pe/sisap/portal2/mayorista/"

FECHA_INICIO_TEXTO = "01/01/2012"
FECHA_FIN_TEXTO = "01/09/2026"

FECHA_INICIO = pd.Timestamp("2012-01-01")
FECHA_FIN = pd.Timestamp("2026-09-01")

MERCADO_GMML = "Gran mercado mayorista de lima"
PRODUCTO_CEBOLLA = "Cebolla"
PRODUCTO_CEBOLLA_ROJA = "Cebolla cabeza roja"

RAW_PRECIO = "Precio Promedio Cebolla Roja GMML 2012-2026.xls"
RAW_VOLUMEN = "Volumen Total Cebolla Roja GMML 2012-2026.xls"

OUTPUT_FILE = OUTPUT_DIR / "dataset_diario_consolidado.csv"
FIGURE_DIR = ROOT / "Figuras" / "experimentacion"
TABLAS_DIR = ROOT / "outputs" / "tablas_experimentacion"


def crear_driver(carpeta_descarga: Path) -> webdriver.Edge:
    carpeta_descarga.mkdir(parents=True, exist_ok=True)

    options = webdriver.EdgeOptions()
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(carpeta_descarga.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )
    options.add_argument("--start-maximized")
    options.add_argument("--disable-popup-blocking")

    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(60)
    return driver


def clic_js(driver: webdriver.Edge, elemento) -> None:
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'});",
        elemento,
    )
    driver.execute_script("arguments[0].click();", elemento)


def colocar_fecha(driver: webdriver.Edge, id_elemento: str, valor: str) -> None:
    elemento = driver.find_element(By.ID, id_elemento)
    driver.execute_script(
        """
        const campo = arguments[0];
        const valor = arguments[1];
        campo.removeAttribute('disabled');
        campo.value = valor;
        campo.dispatchEvent(new Event('input', {bubbles: true}));
        campo.dispatchEvent(new Event('change', {bubbles: true}));
        campo.dispatchEvent(new Event('blur', {bubbles: true}));
        """,
        elemento,
        valor,
    )


def seleccionar_variable(driver: webdriver.Edge, nombre_variable: str) -> None:
    selector = Select(driver.find_element(By.ID, "variables"))
    selector.deselect_all()
    selector.select_by_visible_text(nombre_variable)

    driver.execute_script(
        "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));",
        driver.find_element(By.ID, "variables"),
    )


def esperar_descarga(
    carpeta: Path,
    archivos_antes: set[Path],
    timeout: int = 180,
) -> Path:
    limite = time.time() + timeout

    while time.time() < limite:
        temporales = (
            list(carpeta.glob("*.crdownload"))
            + list(carpeta.glob("*.tmp"))
        )
        actuales = {
            p.resolve()
            for p in carpeta.glob("*.xls")
            if p.is_file()
        }
        nuevos = actuales - archivos_antes

        if nuevos and not temporales:
            candidato = max(
                nuevos,
                key=lambda p: p.stat().st_mtime,
            )
            if candidato.stat().st_size > 0:
                return candidato

        time.sleep(0.5)

    raise TimeoutException(
        "No se completó la descarga del reporte de SISAP dentro del tiempo esperado."
    )


def volver_pestana_principal(
    driver: webdriver.Edge,
    principal: str,
) -> None:
    if principal in driver.window_handles:
        driver.switch_to.window(principal)


def exportar_reporte(
    driver: webdriver.Edge,
    wait: WebDriverWait,
    nombre_destino: str,
) -> Path:
    carpeta = DOWNLOAD_DIR.resolve()
    destino = carpeta / nombre_destino

    if destino.exists():
        destino.unlink()

    archivos_antes = {
        p.resolve()
        for p in carpeta.glob("*.xls")
        if p.is_file()
    }
    principal = driver.current_window_handle

    herramientas = wait.until(
        lambda d: d.find_element(By.ID, "herramientasBoton")
    )
    clic_js(driver, herramientas)

    exportar = wait.until(
        lambda d: d.find_element(By.ID, "botonExportarExcel")
    )
    clic_js(driver, exportar)

    descargado = esperar_descarga(carpeta, archivos_antes)
    descargado.replace(destino)

    time.sleep(0.5)
    volver_pestana_principal(driver, principal)

    return destino


def consultar_y_descargar(
    driver: webdriver.Edge,
    wait: WebDriverWait,
    nombre_variable: str,
    texto_reporte: str,
    nombre_raw: str,
) -> Path:
    seleccionar_variable(driver, nombre_variable)

    boton_consultar = wait.until(
        lambda d: d.find_element(By.ID, "consultar")
    )
    clic_js(driver, boton_consultar)

    wait.until(
        lambda d: d.find_elements(By.CSS_SELECTOR, "#reporte table")
        and texto_reporte.lower()
        in d.find_element(By.ID, "reporte").text.lower()
    )

    return exportar_reporte(
        driver,
        wait,
        nombre_raw,
    )


def configurar_consulta(
    driver: webdriver.Edge,
    wait: WebDriverWait,
) -> None:
    driver.get(URL_SISAP)

    wait.until(
        lambda d: d.find_element(By.ID, "mercado")
    )
    Select(
        driver.find_element(By.ID, "mercado")
    ).select_by_visible_text(MERCADO_GMML)

    boton_expandir_cebolla = wait.until(
        lambda d: d.find_element(
            By.XPATH,
            f"//label[normalize-space(string(.))='{PRODUCTO_CEBOLLA}']"
            "/following-sibling::a[@name='expandir']",
        )
    )
    clic_js(driver, boton_expandir_cebolla)

    cebolla_roja = wait.until(
        lambda d: d.find_element(
            By.XPATH,
            f"//label[normalize-space(string(.))='{PRODUCTO_CEBOLLA_ROJA}']"
            "/input[@type='checkbox']",
        )
    )
    if not cebolla_roja.is_selected():
        clic_js(driver, cebolla_roja)

    wait.until(
        lambda d: d.find_element(By.ID, "intervalo")
    )
    clic_js(
        driver,
        driver.find_element(By.ID, "intervalo"),
    )

    wait.until(
        lambda d: d.find_element(By.ID, "desde").is_enabled()
    )
    colocar_fecha(
        driver,
        "desde",
        FECHA_INICIO_TEXTO,
    )
    colocar_fecha(
        driver,
        "hasta",
        FECHA_FIN_TEXTO,
    )

    driver.execute_script(
        """
        const periodicidad = document.getElementById('periodicidad');
        if (periodicidad) periodicidad.value = 'intervalo';
        """
    )


def _interpretar_numero_sisap(valor: object, variable: str, fila: int) -> float:
    """Conserva el marcador 0 t de SISAP sólo para el abastecimiento."""
    if pd.isna(valor):
        raise ValueError(f"SISAP: celda vacía en {variable}, fila {fila}; comprobar manualmente el reporte.")
    texto = str(valor).strip()
    if texto == "...":
        if variable == "volumen_toneladas":
            return 0.0
        raise ValueError(f"Marcador '...' inesperado en PRECIO, fila {fila}.")
    if re.fullmatch(r"[+]?[0-9]+(?:\.[0-9]+)?", texto) is None:
        raise ValueError(f"Valor no numérico o ambiguo en {variable}, fila {fila}: {texto!r}")
    return float(texto)


def cargar_serie_sisap(path: Path, variable: str) -> pd.DataFrame:
    """No oculta fallos de extracción como ausencias del calendario."""
    tablas = pd.read_html(path, encoding="latin1")
    if not tablas or tablas[0].shape[1] < 2:
        raise ValueError(f"Estructura de reporte SISAP inválida en {path.name}.")
    df = tablas[0].iloc[2:, :2].copy()
    df.columns = ["fecha", variable]
    df["_fila_original"] = df.index + 1
    fechas_texto = df["fecha"].astype("string").str.strip()
    fechas = pd.to_datetime(fechas_texto, format="%d/%m/%Y", errors="coerce")

    footer = fechas.isna() & fechas_texto.str.contains(
        r"^(?:total|promedio|media|fuente|nota)(?:\s|:|$)",case=False,regex=True,na=False
    )
    if footer.any():
        detalle = df.loc[footer, ["_fila_original", "fecha"]].to_dict("records")
        print(f"Pie del reporte excluido de {path.name}: {detalle}",flush=True)
    unknown = fechas.isna() & ~footer
    if unknown.any():
        error = df.loc[unknown, ["_fila_original", "fecha"]].head(10).to_dict("records")
        raise ValueError(f"Fechas inválidas SIN justificar en {path.name}: {error}")
    df = df.loc[~footer].copy()
    df["fecha"] = fechas.loc[~footer].dt.normalize()
    df[variable] = [
        _interpretar_numero_sisap(value, variable, int(row))
        for value,row in zip(df[variable],df["_fila_original"])
    ]
    if ((df[variable] <= 0) if variable == "precio_promedio_soles_kg"
        else (df[variable] < 0)).any():
        raise ValueError(f"Valores fuera de rango en reporte de {variable}.")
    if not df["fecha"].between(FECHA_INICIO, FECHA_FIN).all():
        raise ValueError(f"El reporte {path.name} contiene fechas fuera del intervalo solicitado.")
    return df[["fecha",variable]].sort_values("fecha").reset_index(drop=True)


def validar_serie(
    df: pd.DataFrame,
    nombre: str,
) -> None:
    duplicados = df["fecha"].duplicated(
        keep=False
    )

    if duplicados.any():
        fechas = (
            df.loc[duplicados, "fecha"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"La serie '{nombre}' contiene fechas duplicadas: {fechas}"
        )


def crear_calendario_diario() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fecha": pd.date_range(
                start=FECHA_INICIO,
                end=FECHA_FIN,
                freq="D",
            )
        }
    )


def integrar_y_exportar(
    raw_precio: Path,
    raw_volumen: Path,
) -> pd.DataFrame:
    precio = cargar_serie_sisap(
        raw_precio,
        "precio_promedio_soles_kg",
    )
    volumen = cargar_serie_sisap(
        raw_volumen,
        "volumen_toneladas",
    )

    validar_serie(
        precio,
        "precio",
    )
    validar_serie(
        volumen,
        "volumen",
    )

    calendario = crear_calendario_diario()

    dataset = (
        calendario
        .merge(
            precio,
            on="fecha",
            how="left",
            validate="one_to_one",
        )
        .merge(
            volumen,
            on="fecha",
            how="left",
            validate="one_to_one",
        )
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    metadata = {
        "fecha_descarga_integracion":datetime.now().isoformat(timespec="seconds"),
        "mercado":MERCADO_GMML,"producto":PRODUCTO_CEBOLLA_ROJA,
        "url_sisap":URL_SISAP,"periodo_inicio":str(FECHA_INICIO.date()),
        "periodo_fin":str(FECHA_FIN.date()),"total_fechas":len(dataset),
        "fuentes": {p.name:{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                             "bytes":p.stat().st_size} for p in (raw_precio,raw_volumen)},
        "marcador_abastecimiento_puntos": "0 toneladas sólo en un registro existente",
    }
    (OUTPUT_DIR/"metadata_adquisicion.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    return dataset



def generar_evidencia_adquisicion(dataset: pd.DataFrame) -> None:
    """Tablas y figura de integridad de SISAP; no completa ni inventa registros."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    columnas = ["precio_promedio_soles_kg", "volumen_toneladas"]
    auditoria = []
    for nombre in columnas:
        x = dataset[nombre]
        auditoria.append({
            "variable": nombre, "periodo_inicio": str(dataset["fecha"].min().date()),
            "periodo_fin": str(dataset["fecha"].max().date()),
            "dias_calendario": len(dataset), "dias_observados": int(x.notna().sum()),
            "dias_faltantes": int(x.isna().sum()),
            "porcentaje_faltante": round(float(x.isna().mean()*100), 4),
            "min_observado": float(x.min()), "media_observada": float(x.mean()),
            "max_observado": float(x.max()),
        })
    pd.DataFrame(auditoria).to_csv(TABLAS_DIR / "Tabla_1_auditoria_adquisicion.csv", index=False, encoding="utf-8-sig")
    faltantes = dataset.loc[dataset[columnas].isna().any(axis=1), ["fecha", *columnas]].copy()
    faltantes["precio_ausente"] = faltantes[columnas[0]].isna().astype(int)
    faltantes["abastecimiento_ausente"] = faltantes[columnas[1]].isna().astype(int)
    faltantes.to_csv(TABLAS_DIR / "Tabla_1b_fechas_ausentes_SISAP.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    for variable in columnas:
        mask = dataset[variable].isna().to_numpy(dtype=bool)
        starts = (mask & ~pd.Series(mask).shift(1,fill_value=False).to_numpy())
        grupos = np.cumsum(starts)
        intervals = []
        for idx in np.unique(grupos[mask]):
            dates = dataset.loc[mask & (grupos==idx),"fecha"]
            intervals.append({"variable":variable,"inicio":dates.min().date().isoformat(),
                              "fin":dates.max().date().isoformat(),"dias":int(len(dates))})
        pd.DataFrame(intervals,columns=["variable","inicio","fin","dias"]).to_csv(
            TABLAS_DIR/f"Tabla_1c_intervalos_{variable}.csv",index=False,encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for ax, col, nombre in zip(axes, columnas, ["Precio", "Abastecimiento"]):
        mask = dataset[col].isna().astype(int).to_numpy()
        ax.plot(dataset["fecha"], mask, marker="|", markersize=7, linewidth=0,
                label="Fecha sin registro", color="#B45309")
        ax.set_yticks([0, 1]); ax.set_ylim(-0.2, 1.3)
        ax.set_ylabel(f"{nombre}\n(falta=1)")
        ax.grid(alpha=.15); ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Disponibilidad diaria de registros originales de SISAP - GMML")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figura_0_disponibilidad_SISAP_original.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Auditoría de adquisición y figura de disponibilidad generadas.", flush=True)


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
    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    driver = crear_driver(DOWNLOAD_DIR)
    wait = WebDriverWait(
        driver,
        40,
    )

    try:
        configurar_consulta(
            driver,
            wait,
        )

        raw_precio = consultar_y_descargar(
            driver,
            wait,
            nombre_variable="Precio Promedio",
            texto_reporte="Precio Promedio",
            nombre_raw=RAW_PRECIO,
        )

        raw_volumen = consultar_y_descargar(
            driver,
            wait,
            nombre_variable="Volumen",
            texto_reporte="Volumen",
            nombre_raw=RAW_VOLUMEN,
        )

    finally:
        driver.quit()

    dataset = integrar_y_exportar(
        raw_precio,
        raw_volumen,
    )
    generar_evidencia_adquisicion(dataset)

    print(
        f"Dataset consolidado: {OUTPUT_FILE}"
    )
    print(
        f"Observaciones: {len(dataset)}"
    )
    print(
        f"Periodo: {dataset['fecha'].min().date()} "
        f"a {dataset['fecha'].max().date()}"
    )
    print(
        "Precio faltante: "
        f"{dataset['precio_promedio_soles_kg'].isna().sum()}"
    )
    print(
        "Abastecimiento faltante: "
        f"{dataset['volumen_toneladas'].isna().sum()}"
    )
    print(
        "Fechas sin precio ni abastecimiento: "
        f"{dataset[['precio_promedio_soles_kg', 'volumen_toneladas']].isna().all(axis=1).sum()}"
    )


if __name__ == "__main__":
    ejecutar_con_log(main, "FASE 1", LOG_FILE)
