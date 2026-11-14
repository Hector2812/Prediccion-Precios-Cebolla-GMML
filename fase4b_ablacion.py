"""Evaluación de escenarios sin historial de precio y explicabilidad correspondiente.

Sólo usa los modelos congelados en Fase 3 y el 20 % de prueba. No selecciona
hiperparámetros con el conjunto final. Las comparaciones son EXPLORATORIAS:
la decisión de introducir esta ablación fue posterior a observar SHAP de prueba.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fase4_evaluacion_comparacion as f4

CALENDARIO = "calendario"
CALENDARIO_ABASTECIMIENTO = "calendario_abastecimiento"
CONFIGURACIONES_ABLACION = (CALENDARIO, CALENDARIO_ABASTECIMIENTO)
CONFIGURACIONES_TODAS = ("historico", "historico_abastecimiento", *CONFIGURACIONES_ABLACION)
MODELOS_ABLACION = ("XGBoost","LSTM","TCN","LSTM-XGBoost")


def instalar_configuraciones() -> None:
    f4.CONFIG_LABELS.update({CALENDARIO:"Calendario",CALENDARIO_ABASTECIMIENTO:"Calendario + abastecimiento"})
    f4.CONFIG_SLUGS.update({CALENDARIO:"calendario",CALENDARIO_ABASTECIMIENTO:"calendario_abastecimiento"})
    old_cols=f4.obtener_columnas_tabulares
    old_windows=f4.construir_ventanas
    old_names=f4.obtener_nombres_caracteristicas_secuenciales

    def cols(df:pd.DataFrame,cfg:str)->list[str]:
        if cfg not in CONFIGURACIONES_ABLACION:
            return old_cols(df,cfg)
        calendar=[c for c in f4.CALENDAR_COLUMNS if c in df.columns]
        if len(calendar)!=len(f4.CALENDAR_COLUMNS):
            raise ValueError("Ablación: faltan variables calendario")
        volume=[c for c in df.columns if c.startswith("abastecimiento_lag_")]
        if cfg==CALENDARIO_ABASTECIMIENTO and not volume:
            raise ValueError("Ablación: faltan rezagos de abastecimiento")
        result=calendar+(volume if cfg==CALENDARIO_ABASTECIMIENTO else [])
        assert not any(c.startswith('precio_') for c in result)
        return result

    def names(cfg:str,usar_calendario:bool=True)->list[str]:
        if cfg not in CONFIGURACIONES_ABLACION:
            return old_names(cfg,usar_calendario=usar_calendario)
        base=["abastecimiento"] if cfg==CALENDARIO_ABASTECIMIENTO else []
        if not usar_calendario and not base:
            raise ValueError("Ablación Calendario requiere variables de calendario")
        return base+list(f4.CALENDAR_COLUMNS if usar_calendario else ())

    def windows(df:pd.DataFrame,indices:np.ndarray,window:int,cfg:str,
                usar_calendario:bool=True,exigir_objetivo:bool=True):
        if cfg not in CONFIGURACIONES_ABLACION:
            return old_windows(df,indices,window,cfg,usar_calendario=usar_calendario,
                               exigir_objetivo=exigir_objetivo)
        if window < 1: raise ValueError("Ventana no positiva")
        components=[]
        if cfg==CALENDARIO_ABASTECIMIENTO:
            components.append(df[f4.VOLUME].to_numpy(dtype=float).reshape(-1,1))
        if usar_calendario:
            components.append(f4.codificar_calendario_ciclico(df[f4.DATE]))
        if not components: raise ValueError("Ablación sin entradas")
        mat=np.column_stack(components);target=df[f4.TARGET].to_numpy(dtype=float)
        XX,yy,rows=[],[],[]
        for i in np.asarray(indices,int):
            start=i-window+1
            if start < 0 or i>=len(df): continue
            x=mat[start:i+1]
            if not np.isfinite(x).all() or (exigir_objetivo and not np.isfinite(target[i])):
                continue
            XX.append(x.astype(np.float32,copy=False));yy.append(float(target[i]));rows.append(int(i))
        if not XX:
            return (np.empty((0,window,mat.shape[1]),np.float32),
                    np.empty((0,),np.float32),np.empty((0,),int))
        result=np.asarray(XX,np.float32),np.asarray(yy,np.float32),np.asarray(rows,int)
        if result[0].shape[2]!=len(names(cfg,usar_calendario)):
            raise AssertionError("Dimensión de variables secuenciales inconsistente")
        return result

    f4.obtener_columnas_tabulares=cols
    f4.obtener_nombres_caracteristicas_secuenciales=names
    f4.construir_ventanas=windows


def _calcular_metricas(preds:pd.DataFrame, escala:float)->pd.DataFrame:
    real=preds["real"].to_numpy(dtype=float);valid=np.isfinite(real)
    if not valid.any(): raise ValueError("No hay objetivos observados para evaluar")
    rows=[]
    for cfg in CONFIGURACIONES_TODAS:
        for model in MODELOS_ABLACION:
            col=f4.clave_modelo(model,cfg)
            if col not in preds:raise ValueError(f"Falta predicción {col}")
            pred=preds[col].to_numpy(dtype=float)
            if not np.isfinite(pred).all():raise ValueError(f"Pronóstico no finito {col}")
            m=f4.calcular_metricas_test(real[valid],pred[valid],escala)
            rows.append({"modelo":model,"configuracion":f4.etiqueta_configuracion(cfg),"configuracion_slug":cfg,
                         **m,"n":int(valid.sum()),"objetivos_ausentes":int((~valid).sum())})
    return pd.DataFrame(rows)


def _tabla_comparaciones(preds:pd.DataFrame,metrics:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    """Diferencias pareadas. Tres familias predefinidas de cuatro contrastes c/u."""
    pairs=[
       ("Retiro del precio sin abastecimiento","historico",CALENDARIO),
       ("Retiro del precio con abastecimiento","historico_abastecimiento",CALENDARIO_ABASTECIMIENTO),
       ("Aporte de abastecimiento sin precio",CALENDARIO,CALENDARIO_ABASTECIMIENTO),
    ]
    y=preds["real"].to_numpy(dtype=float)
    dm_rows=[];changes=[]
    for family,a,b in pairs:
        temp=[]
        for model in MODELOS_ABLACION:
            pa=preds[f4.clave_modelo(model,a)].to_numpy(dtype=float)
            pb=preds[f4.clave_modelo(model,b)].to_numpy(dtype=float)
            dm=f4.contraste_diebold_mariano_hac(y,pa,pb)
            record={"familia":family,"modelo":model,"configuracion_A":f4.etiqueta_configuracion(a),
                    "configuracion_B":f4.etiqueta_configuracion(b),**dm}
            temp.append(record)
            met_a=metrics[(metrics.modelo==model)&(metrics.configuracion_slug==a)].iloc[0]
            met_b=metrics[(metrics.modelo==model)&(metrics.configuracion_slug==b)].iloc[0]
            changes.append({"familia":family,"modelo":model,
                            "configuracion_A":f4.etiqueta_configuracion(a),"configuracion_B":f4.etiqueta_configuracion(b),
                            **{f"delta_{m}_B_menos_A":float(met_b[m]-met_a[m]) for m in ("smape","mae","rmse","mase","r2")}})
        dm_rows.append(f4.aplicar_holm(pd.DataFrame(temp)))
    return pd.concat(dm_rows,ignore_index=True),pd.DataFrame(changes)


def _guardar_figura(fig:plt.Figure,name:str):
    f4.FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    path=f4.FIGURE_DIR/name
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    return path


def generar_figuras_ablacion(metrics:pd.DataFrame,changes:pd.DataFrame,dm:pd.DataFrame) -> None:
    """Sólo tres figuras interpretables, evitando duplicar el catálogo previo."""
    modelos=list(MODELOS_ABLACION)
    etiquetas=[f4.etiqueta_configuracion(x) for x in CONFIGURACIONES_TODAS]
    colores=["#245789","#6F9AC1","#AE7428","#68A58B"]
    data=metrics.pivot(index="modelo",columns="configuracion",values="smape").reindex(index=modelos,columns=etiquetas)
    if data.isna().any().any():raise ValueError("Matriz incompleta en figura de ablación")
    fig,ax=plt.subplots(figsize=(10.4,5.2))
    x=np.arange(len(modelos));width=.19
    for j,(cfg,color) in enumerate(zip(etiquetas,colores)):
        ax.bar(x+(j-1.5)*width,data[cfg].to_numpy(dtype=float),width=width,color=color,label=cfg)
    ax.set_xticks(x,modelos);ax.set_ylabel("sMAPE en prueba final (%)")
    ax.set_title("Desempeño con y sin historial del precio - evaluación final")
    ax.legend(fontsize=8,ncol=2);ax.grid(axis="y",alpha=.15)
    fig.tight_layout();_guardar_figura(fig,"Figura_6_11_ablacion_smape_cuatro_configuraciones.png")

    order=["Retiro del precio sin abastecimiento","Retiro del precio con abastecimiento","Aporte de abastecimiento sin precio"]
    fig,axes=plt.subplots(1,3,figsize=(16,5.0),sharey=True,constrained_layout=True)
    lim=max(abs(changes["delta_smape_B_menos_A"]).max()*1.15,.02)
    for ax,fam in zip(axes,order):
        sub=changes[changes.familia.eq(fam)].set_index("modelo").reindex(modelos)
        values=sub["delta_smape_B_menos_A"].to_numpy(dtype=float)
        col=["#ba6a44" if z>0 else "#287e68" for z in values]
        ax.barh(np.arange(len(modelos)),values,color=col,alpha=.9)
        ax.axvline(0,color="black",lw=.8)
        ax.set_yticks(np.arange(len(modelos)),modelos)
        ax.invert_yaxis();ax.set_xlim(-lim,lim)
        ax.set_title(fam,fontsize=10);ax.grid(axis="x",alpha=.15)
        ax.set_xlabel("Cambio sMAPE B - A (p.p.)")
    fig.suptitle("Ablación - cambio de error al retirar o añadir información",fontsize=12)
    _guardar_figura(fig,"Figura_6_12_ablacion_deltas_smape.png")

    fig,axes=plt.subplots(1,3,figsize=(16,5.0),sharey=True,constrained_layout=True)
    bound=max(abs(dm["diferencial_perdida_medio_A_menos_B"]).max()*1.3,1e-5)
    for ax,fam in zip(axes,order):
        part=dm[dm.familia.eq(fam)].set_index("modelo").reindex(modelos)
        values=part["diferencial_perdida_medio_A_menos_B"].to_numpy(dtype=float)
        sig=part["rechaza_h0_holm"].to_numpy(dtype=bool)
        colors=["#b45346" if x>0 else "#287e68" for x in values]
        bars=ax.barh(np.arange(len(modelos)),values,color=colors)
        for bar,s in zip(bars,sig):bar.set_alpha(1 if s else .42)
        ax.axvline(0,color="black",lw=.8);ax.set_xlim(-bound,bound)
        ax.set_yticks(np.arange(len(modelos)),modelos);ax.invert_yaxis()
        ax.set_title(fam,fontsize=10);ax.grid(axis="x",alpha=.16)
        ax.set_xlabel("Pérdida cuadrática media A - B (S/kg)^2")
        for i,(p,s) in enumerate(zip(part["p_valor_holm"],sig)):
            ax.text(.97,i,f"p={p:.3g}"+(" *" if s else ""),fontsize=8,va='center',ha='right',transform=ax.get_yaxis_transform())
    fig.suptitle("Diebold-Mariano - ablación con Holm por familia (* p < 0,05)",fontsize=12)
    _guardar_figura(fig,"Figura_6_13_ablacion_DM_Holm.png")


def generar_importancias_grupo() -> None:
    """SHAP global resumido por grupo dentro de cada modelo, no causalidad."""
    rows=[]
    for cfg in CONFIGURACIONES_TODAS:
        path=f4.EXPLAIN_DIR/f"xgboost_{cfg}_treeshap.csv"
        if not path.exists(): raise FileNotFoundError(f"No está TreeSHAP global para {cfg}: {path}")
        data=pd.read_csv(path)
        if not {"predictor","shap_abs_medio"}.issubset(data):raise ValueError(f"Tabla SHAP incompatible: {path}")
        total=float(data["shap_abs_medio"].sum())
        for group in ("Precio","Calendario","Abastecimiento"):
            sub=data[data.predictor.map(lambda c: ("Precio" if str(c).startswith('precio_') else
                       "Abastecimiento" if str(c).startswith('abastecimiento_') else
                       "Calendario" if c in f4.CALENDAR_COLUMNS else "Otro")==group)]
            attr=float(sub["shap_abs_medio"].sum())
            rows.append({"configuracion":f4.etiqueta_configuracion(cfg),"configuracion_slug":cfg,"grupo":group,
                         "suma_shap_abs_medio_S_kg":attr,
                         "porcentaje_shap_abs_dentro_modelo":100*attr/total if total>0 else 0.,
                         "n_variables":len(sub)})
    tabla=pd.DataFrame(rows)
    tabla.to_csv(f4.TABLE_DIR/"Tabla_8b_TreeSHAP_por_grupos_ablacion.csv",index=False,encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(10.6,4.7))
    labels=[f4.etiqueta_configuracion(x) for x in CONFIGURACIONES_TODAS]
    x=np.arange(len(labels));bottom=np.zeros(len(labels))
    for group,color in [("Precio","#23639a"),("Calendario","#b0772b"),("Abastecimiento","#308471")]:
        values=tabla[tabla.grupo.eq(group)].set_index("configuracion_slug").reindex(CONFIGURACIONES_TODAS)["porcentaje_shap_abs_dentro_modelo"].to_numpy(dtype=float)
        ax.bar(x,values,bottom=bottom,label=group,color=color)
        bottom+=values
    ax.set_xticks(x,labels,rotation=12,ha='right')
    ax.set_ylabel("Participación en suma de |TreeSHAP| dentro de cada modelo (%)")
    ax.set_ylim(0,105);ax.set_title("XGBoost - atribuciones globales por grupo de predictores")
    ax.legend(fontsize=8,ncol=3);ax.grid(axis="y",alpha=.15)
    fig.tight_layout();_guardar_figura(fig,"Figura_6_14_TreeSHAP_por_grupos_ablacion.png")
    return tabla


def ejecutar_ablacion(prediction_table:pd.DataFrame,df:pd.DataFrame,
                       implementation_idx:np.ndarray,test_idx:np.ndarray,
                       artefactos:dict,escala_mase:float) -> None:
    instalar_configuraciones()
    f4.imprimir_titulo("FASE 4 - ABLACIÓN Y EXPLICABILIDAD SIN HISTORIAL DEL PRECIO")
    state=f4.VALIDATION_DIR/"ablacion_estado_fase3.json"
    if not state.exists() or json.loads(state.read_text(encoding="utf-8")).get("estado")!="COMPLETADA":
        raise RuntimeError("La Fase 3 no completó los modelos de ablación.")
    expected_hash=json.loads(state.read_text(encoding="utf-8"))["sha256_dataset"]
    if hashlib.sha256(f4.INPUT_FILE.read_bytes()).hexdigest()!=expected_hash:
        raise RuntimeError("Se mezclaron datasets entre el entrenamiento y la ablación.")
    resources={}
    test_y=prediction_table["real"].to_numpy(dtype=float)
    target_dates=prediction_table["fecha_target"].to_numpy()
    for cfg in CONFIGURACIONES_ABLACION:
        label=f4.etiqueta_configuracion(cfg)
        f4.imprimir_subtitulo(f"Evaluación sin precio - {label}")
        xgb,info=f4.cargar_xgboost(artefactos,cfg)
        xpred,xmat,cols=f4.predecir_xgboost_test(xgb,df,test_idx,cfg)
        if any(c.startswith("precio_") for c in cols):raise AssertionError("XGBoost reintroduce el precio")
        prediction_table[f4.clave_modelo("XGBoost",cfg)]=xpred
        resources[f4.clave_modelo("XGBoost",cfg)]={"model":xgb,"X":xmat,"cols":cols,"pred":xpred}

        lstm,lstm_params,lstm_scalers=f4.cargar_lstm(artefactos,cfg)
        lstm_pred,lstm_test=f4.predecir_lstm_test(lstm,lstm_params,lstm_scalers,df,test_idx,cfg)
        prediction_table[f4.clave_modelo("LSTM",cfg)]=lstm_pred
        X_lstm,y_lstm,_=f4.construir_ventanas(df,implementation_idx,int(lstm_params["window"]),cfg)
        X_lstm_scaled,_=f4.escalar_secuencias(X_lstm,y_lstm,lstm_scalers["x_scaler"],lstm_scalers["y_scaler"])
        resources[f4.clave_modelo("LSTM",cfg)]={"model":lstm,"params":lstm_params,"scalers":lstm_scalers,
                                                 "test":lstm_test,"train_scaled":X_lstm_scaled}

        tcn,tcn_params,tcn_scalers=f4.cargar_tcn(artefactos,cfg)
        tcn_pred,tcn_test=f4.predecir_tcn_test(tcn,tcn_params,tcn_scalers,df,test_idx,cfg)
        prediction_table[f4.clave_modelo("TCN",cfg)]=tcn_pred
        X_tcn,y_tcn,_=f4.construir_ventanas(df,implementation_idx,int(tcn_params["window"]),cfg,
                                           usar_calendario=bool(tcn_params.get("usar_calendario",True)))
        X_tcn_scaled,_=f4.escalar_secuencias(X_tcn,y_tcn,tcn_scalers["x_scaler"],tcn_scalers["y_scaler"])
        resources[f4.clave_modelo("TCN",cfg)]={"model":tcn,"params":tcn_params,"scalers":tcn_scalers,
                                                "test":tcn_test,"train_scaled":X_tcn_scaled}

        residual,rinfo=f4.cargar_corrector_hibrido(artefactos,cfg)
        hybrid_pred,xres,rcols=f4.predecir_hibrido_test(residual,df,test_idx,cfg,lstm_pred)
        if any(c.startswith('precio_') for c in rcols):raise AssertionError("El corrector reintroduce precio")
        prediction_table[f4.clave_modelo("LSTM-XGBoost",cfg)]=hybrid_pred
        resources[f4.clave_modelo("LSTM-XGBoost",cfg)]={"model":residual,"X":xres,"cols":rcols,
                                                       "pred":hybrid_pred,"base":lstm_pred}
        print(f"{label}: cuatro modelos evaluados en {len(test_idx)} fechas; sin precio como predictor.",flush=True)

    pred_path=f4.OUTPUT_DIR/"predicciones_evaluacion_final.csv"
    prediction_table.to_csv(pred_path,index=False,encoding="utf-8-sig",date_format="%Y-%m-%d")
    metrics=_calcular_metricas(prediction_table,escala_mase)
    metrics.to_csv(f4.OUTPUT_DIR/"metricas_ablacion_cuatro_configuraciones.csv",index=False,encoding="utf-8-sig")
    metrics.to_csv(f4.TABLE_DIR/"Tabla_8_metricas_ablacion.csv",index=False,encoding="utf-8-sig")
    dm,changes=_tabla_comparaciones(prediction_table,metrics)
    dm.to_csv(f4.OUTPUT_DIR/"dm_ablacion_Holm.csv",index=False,encoding="utf-8-sig")
    dm.to_csv(f4.TABLE_DIR/"Tabla_9_DM_Holm_ablacion.csv",index=False,encoding="utf-8-sig")
    changes.to_csv(f4.TABLE_DIR/"Tabla_8a_cambios_ablacion.csv",index=False,encoding="utf-8-sig")
    generar_figuras_ablacion(metrics,changes,dm)

    status={}
    for cfg in CONFIGURACIONES_ABLACION:
        label=f4.etiqueta_configuracion(cfg);slug=f4.slug_configuracion(cfg)
        print(f"Explicabilidad sin precio - {label}",flush=True)
        x=resources[f4.clave_modelo("XGBoost",cfg)]
        f4.explicar_xgboost_treeshap(x["model"],x["X"],x["cols"],f4.EXPLAIN_DIR/f"xgboost_{slug}_treeshap.csv")
        f4.generar_figuras_treeshap(x["model"],x["X"],x["cols"],f"xgboost_{slug}",target_dates,test_y,x["pred"])
        status[f"XGBoost_{slug}"]="TreeSHAP global y local"

        l=resources[f4.clave_modelo("LSTM",cfg)]
        if f4.OMITIR_TIMESHAP:
            status[f"LSTM_{slug}"]="TimeSHAP omitido por configuración explícita"
        else:
            f4.explicar_lstm_timeshap(l["model"],l["train_scaled"],l["test"]["X_scaled"],
                                      f4.obtener_nombres_caracteristicas_secuenciales(cfg),l["scalers"]["y_scaler"],
                                      target_dates,f4.EXPLAIN_DIR/f"lstm_{slug}_timeshap")
            f4.registrar_componente_lstm_hibrido(cfg)
            f4.generar_figuras_timeshap(f4.EXPLAIN_DIR/f"lstm_{slug}_timeshap",slug)
            status[f"LSTM_{slug}"]="TimeSHAP global y local"

        t=resources[f4.clave_modelo("TCN",cfg)]
        f4.explicar_tcn_integrated_gradients(t["model"],t["train_scaled"],t["test"]["X_scaled"],
               f4.obtener_nombres_caracteristicas_secuenciales(cfg,
                  usar_calendario=bool(t["params"].get("usar_calendario",True))),
               t["scalers"]["y_scaler"],target_dates,
               f4.EXPLAIN_DIR/f"tcn_{slug}_integrated_gradients")
        f4.generar_figuras_tcn_ig(f4.EXPLAIN_DIR/f"tcn_{slug}_integrated_gradients",slug)
        status[f"TCN_{slug}"]="Integrated Gradients global y local"

        h=resources[f4.clave_modelo("LSTM-XGBoost",cfg)]
        f4.explicar_xgboost_treeshap(h["model"],h["X"],h["cols"],
             f4.EXPLAIN_DIR/f"hibrido_xgboost_{slug}_treeshap.csv")
        f4.generar_figuras_treeshap(h["model"],h["X"],h["cols"],f"hibrido_corrector_{slug}",
             target_dates,test_y-h["base"],h["pred"]-h["base"],residuo=True,base_lstm=h["base"])
        status[f"LSTM-XGBoost_{slug}"]="TreeSHAP sólo del corrector residual"

    generar_importancias_grupo()
    f4.guardar_json(f4.OUTPUT_DIR/"ablacion_estado_fase4.json",{
        "estado":"COMPLETADA","fin":datetime.now().isoformat(timespec="seconds"),
        "configuraciones":[f4.etiqueta_configuracion(x) for x in CONFIGURACIONES_TODAS],
        "modelos_sin_precio":list(MODELOS_ABLACION),
        "familias_Holm":["Retiro del precio sin abastecimiento","Retiro del precio con abastecimiento","Aporte de abastecimiento sin precio"],
        "contrastes_por_familia":4,"exploratorio_post_hoc":True,
        "shap_no_equivale_a_efecto_causal":True,
        "explicabilidad":status,
    })
    print("Ablación, DM-Holm, TreeSHAP, TimeSHAP e Integrated Gradients guardados.",flush=True)


if __name__ == "__main__":
    raise SystemExit("Ejecute fase4_evaluacion_comparacion.py, que integra esta evaluación después de la evaluación principal")
