# src/optimization.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .conf import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary
from .loader import guardar_iteracion

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    params = {
        'objective':'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
    # Completar a gusto!!!!!!!
    'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
    'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1], log=True),
    'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
    'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
    'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGB['min_child_samples'][0], PARAMETROS_LGB['min_child_samples'][1]),
    'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGB['max_depth'][0], PARAMETROS_LGB['max_depth'][1]),
    'reg_alpha': trial.suggest_float('reg_alpha', PARAMETROS_LGB['reg_alpha'][0], PARAMETROS_LGB['reg_alpha'][1]),
    'reg_lambda': trial.suggest_float('reg_lambda', PARAMETROS_LGB['reg_lambda'][0], PARAMETROS_LGB['reg_lambda'][1]),
    'min_gain_to_split': 0.0,   # Penetrar splits con ganancia mínima
    'verbose': -1,              # Reducir verbosidad
    'verbosity': -1,            # Silenciar mensajes adicionales
    'silent': True,             # Modo silencioso
    'bin': 31,
    'random_state': SEMILLA[0],  # Desde configuración YAML
    }
    

    # Preparar datos usando configuración YAML
    if isinstance(MES_TRAIN, list):
        df_train = df[df["foto_mes"].isin(MES_TRAIN)]
    else:
        df_train = df[df["foto_mes"] == MES_TRAIN]

    df_val = df[df["foto_mes"] == MES_VALIDACION]

    # Usar target (clase_ternaria ya convertida a binaria)
    y_train = df_train["clase_ternaria"].values
    y_val = df_val["clase_ternaria"].values

    # Features: usar todas las columnas excepto target
    X_train = df_train.drop(columns=["clase_ternaria"])
    X_val = df_val.drop(columns=["clase_ternaria"])

    # Entrenar modelo con función de ganancia personalizada
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    feval=ganancia_lgb_binary,  # Función de ganancia personalizada
    callbacks=[lgb.early_stopping(stopping_rounds=(50), verbose=False),
                lgb.log_evaluation(0)]
    )

    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)
    y_pred_binary = (y_pred_proba > 0.0251).astype(int)  # Usar mismo umbral que en ganancia_lgb_binary

    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)

    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:.0f}")

    return ganancia_total




def optimizar(df, n_trial=100) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trial} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Completar!!!!!!!!
  
    study = optuna.create_study(
        direction='maximize',
        study_name = study_name
    )

    objective_with_data = lambda trial: objetivo_ganancia(trial,df)

    study.optimize(objective_with_data,n_trial,show_progress_bar=True)
    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
  
    return study




def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")

    # Configurar parámetros del modelo
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        'boosting': 'gbdt',
        'num_threads': -1,
        **mejores_params
       
    }
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]

    # Target y Features
    X_train = df_train_completo.drop(['clase_ternaria'], axis=1)
    y_train = df_train_completo['clase_ternaria']
    X_test = df_test.drop(['clase_ternaria'], axis=1)
    y_test = df_test['clase_ternaria']


    # Crear LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)


    modelo = lgb.train(
            params,
            train_data,
            num_boost_round=mejores_params.get('num_iterations', 1000)
        )

    # Predecir y calcular ganancia
    y_pred_proba = modelo.predict(X_test)
    y_pred_binary = (y_pred_proba > 0.0251).astype(int)
  
    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary)
  
    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas)
    }
  
    return resultados

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    # Guarda en resultados/{STUDY_NAME}_test_results.json
    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesiana


