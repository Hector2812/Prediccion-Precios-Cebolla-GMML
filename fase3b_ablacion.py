"""Ablación por grupos para XGBoost, LSTM, TCN y LSTM-XGBoost.

Se importa la implementación original para reutilizar EXACTAMENTE el mismo
TimeSeriesSplit, Optuna, escalado, arquitecturas y lógica OOF del híbrido.
Nunca se utilizan precios como predictores en las dos configuraciones nuevas.
El precio objetivo real t+1 sigue siendo la etiqueta supervisada.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import fase3_implementacion_modelos as f3

CALENDARIO = "calendario"
CALENDARIO_ABASTECIMIENTO = "calendario_abastecimiento"
CONFIGURACIONES_ABLACION = (CALENDARIO, CALENDARIO_ABASTECIMIENTO)
MODELOS_ABLACION = ("XGBoost", "LSTM", "TCN", "LSTM-XGBoost")


def instalar_configuraciones() -> None:
    """Habilita los dos escenarios adicionales sin tocar el flujo principal."""
    f3.CONFIG_LABELS.update({
        CALENDARIO: "Calendario",
        CALENDARIO_ABASTECIMIENTO: "Calendario + abastecimiento",
    })
    f3.CONFIG_SLUGS.update({
        CALENDARIO: "calendario",
        CALENDARIO_ABASTECIMIENTO: "calendario_abastecimiento",
    })
    columnas_originales = f3.obtener_columnas_tabulares
    ventanas_originales = f3.construir_ventanas
    nombres_originales = f3.obtener_nombres_caracteristicas_secuenciales

    def columnas(df: pd.DataFrame, configuracion: str) -> list[str]:
        if configuracion not in CONFIGURACIONES_ABLACION:
            return columnas_originales(df, configuracion)
        cal = [c for c in f3.CALENDAR_COLUMNS if c in df.columns]
        if len(cal) != len(f3.CALENDAR_COLUMNS):
            raise ValueError("Faltan columnas de calendario de fecha objetivo t+1.")
        volumen = [c for c in df.columns if c.startswith("abastecimiento_lag_")]
        if configuracion == CALENDARIO_ABASTECIMIENTO and not volumen:
            raise ValueError("No existen rezagos de abastecimiento para la ablación.")
        resultado = cal + (volumen if configuracion == CALENDARIO_ABASTECIMIENTO else [])
        if any(c.startswith("precio_") for c in resultado):
            raise AssertionError("Fuga de precio: columnas tabulares de ablación.")
        return resultado

    def nombres(configuracion: str, usar_calendario: bool = True) -> list[str]:
        if configuracion not in CONFIGURACIONES_ABLACION:
            return nombres_originales(configuracion, usar_calendario=usar_calendario)
        nombre_base = (["abastecimiento"] if configuracion == CALENDARIO_ABASTECIMIENTO else [])
        if not usar_calendario and not nombre_base:
            raise ValueError("Calendario sin calendario dejaría el modelo sin entradas.")
        return [*nombre_base, *(f3.CALENDAR_COLUMNS if usar_calendario else ())]

    def ventanas(df: pd.DataFrame, indices: np.ndarray, window: int,
                 configuracion: str, usar_calendario: bool = True):
        if configuracion not in CONFIGURACIONES_ABLACION:
            return ventanas_originales(df, indices, window, configuracion,
                                       usar_calendario=usar_calendario)
        if window < 1:
            raise ValueError("La ventana debe ser positiva.")
        variables = []
        if configuracion == CALENDARIO_ABASTECIMIENTO:
            variables.append(df[f3.VOLUME].to_numpy(dtype=float).reshape(-1, 1))
        if usar_calendario:
            # Mantiene la misma codificación histórica utilizada por las redes originales.
            variables.append(f3.codificar_calendario_ciclico(df[f3.DATE]))
        if not variables:
            raise ValueError("No existe ninguna entrada para las ventanas.")
        matriz = np.column_stack(variables)
        target = df[f3.TARGET].to_numpy(dtype=float)
        X, y, rows = [], [], []
        for i in np.asarray(indices, dtype=int):
            start = i - window + 1
            if start < 0 or i >= len(df):
                continue
            x = matriz[start:i + 1]
            if not np.isfinite(x).all() or not np.isfinite(target[i]):
                continue
            X.append(x.astype(np.float32, copy=False))
            y.append(float(target[i]))
            rows.append(int(i))
        n_features = matriz.shape[1]
        if not X:
            return (np.empty((0,window,n_features),np.float32),
                    np.empty((0,),np.float32),np.empty((0,),int))
        output = (np.asarray(X,np.float32), np.asarray(y,np.float32), np.asarray(rows,int))
        if output[0].shape[2] != len(nombres(configuracion,usar_calendario)):
            raise AssertionError("Incompatibilidad de columnas secuenciales.")
        return output

    f3.obtener_columnas_tabulares = columnas
    f3.obtener_nombres_caracteristicas_secuenciales = nombres
    f3.construir_ventanas = ventanas


def verificar_ablacion(df: pd.DataFrame, indices_desarrollo: np.ndarray,
                       test_idx: np.ndarray, folds) -> None:
    """Pruebas previas al entrenamiento: etiquetas, cortes y no filtración de precio."""
    for cfg in CONFIGURACIONES_ABLACION:
        columnas = f3.obtener_columnas_tabulares(df, cfg)
        assert columnas and not any(c.startswith('precio_') for c in columnas)
        for window in (7, 15, 30, 60):
            X, y, rows = f3.construir_ventanas(df, test_idx, window, cfg)
            validos = np.asarray([i for i in test_idx if np.isfinite(df.loc[i,f3.TARGET])],int)
            if not np.array_equal(rows,validos):
                raise AssertionError(f"Fechas de prueba diferentes para {cfg}, ventana {window}.")
            if X.shape[-1] != len(f3.obtener_nombres_caracteristicas_secuenciales(cfg)):
                raise AssertionError("Dimensión de secuencia inconsistente")
            if len(rows) and not np.allclose(y,df.loc[rows,f3.TARGET].to_numpy(dtype=float),rtol=1e-6,atol=1e-6):
                raise AssertionError("El objetivo fue modificado en la ablación")
        for numero,(ent, val) in enumerate(folds,1):
            if len(ent)==0 or len(val)==0 or max(ent)>=min(val):
                raise AssertionError(f"Orden temporal erróneo en fold {numero}")
            _,_,rows,_=f3.preparar_tabular(df,val,cfg)
            validos=np.asarray([i for i in val if np.isfinite(df.loc[i,f3.TARGET])],int)
            if not np.array_equal(rows,validos):
                raise AssertionError(f"Diferencia de fechas de CV {cfg} fold={numero}")
    f3.TABLE_DIR.mkdir(parents=True,exist_ok=True)
    tabla=[]
    for cfg in CONFIGURACIONES_ABLACION:
        for col in f3.obtener_columnas_tabulares(df,cfg):
            tabla.append({"configuracion":f3.etiqueta_configuracion(cfg),"predictor":col,"grupo":"calendario" if col in f3.CALENDAR_COLUMNS else "abastecimiento"})
    pd.DataFrame(tabla).to_csv(f3.TABLE_DIR/"Tabla_3e_entradas_ablacion.csv",index=False,encoding="utf-8-sig")


def ejecutar_ablacion() -> None:
    """Se llama automáticamente después del entrenamiento principal de Fase 3."""
    instalar_configuraciones()
    f3.imprimir_titulo("FASE 3 - ABLACIÓN POR GRUPOS DE PREDICTORES")
    df=f3.cargar_dataset()
    implementacion, prueba, meta=f3.separar_implementacion_prueba(df)
    folds=f3.construir_folds_temporales(implementacion)
    verificar_ablacion(df,implementacion,prueba,folds)
    mejores = json.loads((f3.OUTPUT_DIR/"mejores_configuraciones.json").read_text(encoding="utf-8"))
    artefactos = json.loads((f3.OUTPUT_DIR/"artefactos_modelos.json").read_text(encoding="utf-8"))
    validaciones=[]

    for cfg in CONFIGURACIONES_ABLACION:
        etiqueta,slug=f3.etiqueta_configuracion(cfg),f3.slug_configuracion(cfg)
        f3.imprimir_titulo(f"ABLACIÓN: {etiqueta}")
        with f3.registrar_bloque(f"XGBoost - {etiqueta} - CV"):
            best_xgb,study_xgb,folds_xgb,_=f3.seleccionar_xgboost(df,folds,cfg)
        f3.guardar_resultados_optuna(study_xgb,f3.OUTPUT_DIR/f"xgboost_busqueda_{slug}.csv",folds_xgb,f3.OUTPUT_DIR/f"xgboost_validacion_{slug}.csv")
        mejores[f3.clave_modelo("XGBoost",cfg)]={**best_xgb,"smape_validacion":float(study_xgb.best_value)}
        artefactos[f3.clave_modelo("XGBoost",cfg)]=f3.entrenar_xgboost_final(df,implementacion,cfg,best_xgb)
        validaciones.append(folds_xgb)

        with f3.registrar_bloque(f"LSTM - {etiqueta} - CV"):
            best_lstm,ep_lstm,study_lstm,folds_lstm,hist_lstm=f3.seleccionar_lstm(df,folds,cfg)
        f3.guardar_resultados_optuna(study_lstm,f3.OUTPUT_DIR/f"lstm_busqueda_{slug}.csv",folds_lstm,f3.OUTPUT_DIR/f"lstm_validacion_{slug}.csv")
        mejores[f3.clave_modelo("LSTM",cfg)]={**best_lstm,"learning_rate":f3.LSTM_LEARNING_RATE,"epocas_reajuste":ep_lstm,"smape_validacion":float(study_lstm.best_value)}
        f3.guardar_historial_red_experimentacion(hist_lstm,"LSTM",cfg)
        artefactos[f3.clave_modelo("LSTM",cfg)]=f3.entrenar_lstm_final(df,implementacion,cfg,best_lstm,ep_lstm)
        validaciones.append(folds_lstm)

        with f3.registrar_bloque(f"TCN - {etiqueta} - CV"):
            best_tcn,ep_tcn,study_tcn,folds_tcn,hist_tcn=f3.seleccionar_tcn(df,folds,cfg)
        f3.guardar_resultados_optuna(study_tcn,f3.OUTPUT_DIR/f"tcn_busqueda_{slug}.csv",folds_tcn,f3.OUTPUT_DIR/f"tcn_validacion_{slug}.csv")
        f3.guardar_historial_red_experimentacion(hist_tcn,"TCN",cfg)
        mejores[f3.clave_modelo("TCN",cfg)]={**best_tcn,"n_capas_convolucionales":2*int(best_tcn["n_blocks"]),
             "dilataciones":[2**i for i in range(int(best_tcn["n_blocks"]))],
             "learning_rate":f3.TCN_LEARNING_RATE,"batch_size":f3.TCN_BATCH_SIZE,
             "max_epochs":f3.TCN_MAX_EPOCHS,"patience":f3.TCN_PATIENCE,
             "variables_calendario":True,"arquitectura_tcn":f3.TCN_ARCHITECTURE_ID,
             "epocas_optimas_folds":[int(v) for v in folds_tcn["epoca_optima"]],
             "epocas_reajuste":int(ep_tcn),"regla_epocas_reajuste":"mediana de best_epoch de los 5 folds del mejor trial",
             "campo_receptivo":f3.calcular_campo_receptivo_tcn(int(best_tcn["n_blocks"]),int(best_tcn["kernel_size"])),
             "smape_validacion":float(study_tcn.best_value)}
        artefactos[f3.clave_modelo("TCN",cfg)]=f3.entrenar_tcn_final(df,implementacion,cfg,best_tcn,ep_tcn,float(study_tcn.best_value))
        validaciones.append(folds_tcn)

        with f3.registrar_bloque(f"LSTM-XGBoost - {etiqueta} - CV OOF y corrector"):
            cache=f3.preparar_cache_hibrido(df,folds,cfg,best_lstm,ep_lstm)
            best_res,study_h,folds_h,_=f3.seleccionar_hibrido(cache,cfg)
        f3.guardar_resultados_optuna(study_h,f3.OUTPUT_DIR/f"hibrido_busqueda_{slug}.csv",folds_h,f3.OUTPUT_DIR/f"hibrido_validacion_{slug}.csv")
        mejores[f3.clave_modelo("LSTM-XGBoost",cfg)]={"lstm":best_lstm,"xgboost_residual":best_res,
           "smape_validacion":float(study_h.best_value),"seleccion_anidada_lstm_por_fold":f3.HYBRID_NESTED_LSTM_SELECTION}
        artefactos[f3.clave_modelo("LSTM-XGBoost",cfg)]=f3.entrenar_hibrido_final(df,implementacion,cfg,best_lstm,ep_lstm,best_res)
        features=artefactos[f3.clave_modelo("LSTM-XGBoost",cfg)].get("columnas_corrector",[])
        if any(str(x).startswith('precio_') for x in features):
            raise AssertionError("El corrector híbrido incorpora precios explícitos")
        validaciones.append(folds_h)
        pd.DataFrame(folds_h).to_csv(f3.OUTPUT_DIR/f"ablacion_resumen_hibrido_{slug}.csv",index=False,encoding="utf-8-sig")
        # Guardado tras cada escenario, de modo que no se pierdan resultados completos.
        f3.guardar_json(f3.OUTPUT_DIR/"mejores_configuraciones.json",mejores)
        f3.guardar_json(f3.OUTPUT_DIR/"artefactos_modelos.json",f3.normalizar_rutas_artefactos(artefactos))
        print(f"ABLACIÓN COMPLETADA: {etiqueta}",flush=True)

    tabla = f3.construir_resumen_validacion(validaciones,pd.DataFrame())
    tabla.to_csv(f3.OUTPUT_DIR/"ablacion_resumen_validacion.csv",index=False,encoding="utf-8-sig")
    tabla.to_csv(f3.TABLE_DIR/"Tabla_3f_validacion_ablacion.csv",index=False,encoding="utf-8-sig")
    generar_figura_validacion_ablacion(tabla)
    f3.guardar_json(f3.OUTPUT_DIR/"ablacion_estado_fase3.json",{
        "estado":"COMPLETADA","fecha":datetime.now().isoformat(timespec='seconds'),
        "escenarios":[f3.etiqueta_configuracion(c) for c in CONFIGURACIONES_ABLACION],
        "modelos":list(MODELOS_ABLACION),"semilla":f3.SEED,
        "protocolo":"mismo 80/20 y TimeSeriesSplit expansivo de cinco folds",
        "prueba_final_utilizada_en_seleccion":False,
        "sha256_dataset":hashlib.sha256(f3.INPUT_FILE.read_bytes()).hexdigest(),
    })


def generar_figura_validacion_ablacion(tabla:pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    if tabla.empty: return
    fig,ax=plt.subplots(figsize=(10.5,5.0))
    order=["XGBoost","LSTM","TCN","LSTM-XGBoost"]
    ys=np.arange(len(order))
    data=tabla.set_index(["modelo","configuracion"])
    for k,(cfg,col) in enumerate([(CALENDARIO,"#23639a"),(CALENDARIO_ABASTECIMIENTO,"#be7a31")]):
        etiqueta=f3.etiqueta_configuracion(cfg)
        valores=[float(data.loc[(m,etiqueta),"smape_promedio"]) for m in order]
        ax.barh(ys+(-.18 if k==0 else .18),valores,height=.32,label=etiqueta,color=col,alpha=.85)
    ax.set_yticks(ys,order);ax.invert_yaxis()
    ax.set(xlabel="sMAPE promedio en validación temporal (%)",title="Modelos sin historial del precio - cinco folds")
    ax.legend(fontsize=8);ax.grid(axis='x',alpha=.17)
    fig.tight_layout()
    f3.FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    fig.savefig(f3.FIGURE_DIR/"Figura_5b_validacion_ablacion.png",dpi=300,bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    ejecutar_ablacion()
