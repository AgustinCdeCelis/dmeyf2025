import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .conf import FINAL_TRAIN, FINAL_PREDIC, SEMILLA, STUDY_NAME
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary
import joblib

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
    # 1. Segmentar DataFrames
    df_train = df[df['foto_mes'].isin(FINAL_TRAIN)].copy()
    df_predict = df[df['foto_mes'] == FINAL_PREDIC].copy()

    # 2. Corroborar que no esten vacios los df (Validación básica)
    if df_train.empty or df_predict.empty:
        raise ValueError("Los DataFrames de entrenamiento o predicción están vacíos. Revise las fechas en conf.yaml")
    # Datos de predicción: período FINAL_PREDICT

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")
  
    #Corroborar que no esten vacios los df

    # Preparar features y target para entrenamiento
  
    X_train = df_train.drop(['clase_ternaria'], axis=1)
    y_train = df_train['clase_ternaria']

    # Preparar features para predicción
    X_predict = df_predict.drop(['clase_ternaria'], axis=1)
    clientes_predict = df_predict['numero_de_cliente']

    logger.info(f"Features utilizadas: {len(X_train.columns)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
  
    return X_train, y_train, X_predict, clientes_predict

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.
  
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
  
    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")
  
    # Configurar parámetros del modelo
    #params = {
    #    'objective': 'binary',
    #    'metric': 'None',  # Usamos nuestra métrica personalizada
    #    'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
    #    'verbose': -1,
    #    **mejores_params  # Agregar los mejores hiperparámetros
    #}
    params = {
        'objective': 'binary',
        'metric': 'auc',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        'boosting': 'gbdt',
        'num_threads': -1,
        **mejores_params
       
    }
  
    logger.info(f"Parámetros del modelo: {params}")
  
    # Crear dataset de LightGBM
    train_data =lgb.Dataset(X_train,label=y_train)
  
    # Entrenar modelo
    logger.info("Entrenando modelo...")
    modelo = lgb.train(
            params,
            train_data,
            num_boost_round=mejores_params.get('num_iterations', 1000)
        )



    return modelo

def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray, umbral: float = 0.025) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.
  
    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria
  
    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")
  
    # Generar probabilidades con el modelo entrenado
    predicciones = modelo.predict(X_predict)
    # Convertir a predicciones binarias con el umbral establecido
    y_pred_binary = (predicciones > umbral).astype(int)
    # Crear DataFrame de 'resultados' con nombres de atributos que pide kaggle
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'predict': y_pred_binary
    })
  
    # Estadísticas de predicciones
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['predict'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")
  
    return resultados #df pd.DataFrame con numero de cliente y predict


def guardar_predicciones_finales(resultados_df: pd.DataFrame, nombre_archivo=None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict.
  
    Args:
        resultados_df: DataFrame con numero_cliente y predict
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)
  
    Returns:
        str: Ruta del archivo guardado
    """
    # Crear carpeta predict si no existe
    os.makedirs("predict", exist_ok=True)
  
    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME
  
    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = f"predict/{nombre_archivo}_{timestamp}.csv"
  
    # Validar formato del DataFrame
  
    # Validar tipos de datos
  
    # Validar valores de predict (deben ser 0 o 1)

  
    # Guardar archivo
    resultados_df.to_csv(ruta_archivo, index=False)
  
    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    logger.info(f"Formato del archivo:")
    logger.info(f"  Columnas: {list(resultados_df.columns)}")
    logger.info(f"  Registros: {len(resultados_df):,}")
    logger.info(f"  Primeras filas:")
    logger.info(f"{resultados_df.head()}")
  
    return ruta_archivo


def guardar_modelo(modelo, periodos_entrenamiento: list | str) -> str:
    """
    Guarda el modelo entrenado en un archivo serializado (.pkl) en la carpeta 'modelos'.

    Args:
        modelo: El objeto del modelo entrenado (ej: LightGBM Booster).
        periodos_entrenamiento: Lista o string con los meses usados para entrenar.

    Returns:
        str: Ruta del archivo del modelo guardado.
    """
    # 1. Crear carpeta si no existe
    os.makedirs("modelos", exist_ok=True)
    
    # 2. Generar nombre de archivo único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Asegurar que el nombre del estudio se usa si está disponible
    nombre_base = STUDY_NAME if 'STUDY_NAME' in locals() or 'STUDY_NAME' in globals() else "modelo_final"
    
    # Usamos los periodos en el nombre
    periodos_str = "_".join([str(p) for p in periodos_entrenamiento])
    
    nombre_archivo = f"{nombre_base}_TRAIN_{periodos_str}_{timestamp}.pkl"
    ruta_archivo = os.path.join("modelos", nombre_archivo)
    
    # 3. Guardar el modelo
    try:
        joblib.dump(modelo, ruta_archivo)
        logger.info(f"Modelo final guardado exitosamente en: {ruta_archivo}")
        return ruta_archivo
    except Exception as e:
        logger.error(f"Error al guardar el modelo: {e}")
        raise
