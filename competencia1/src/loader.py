import pandas as pd
import logging
from datetime import datetime
import os
import json
from src.conf import STUDY_NAME, SEMILLA, MES_TRAIN, MES_VALIDACION



logger = logging.getLogger("__name__")


def cargar_datos(path:str) -> pd.DataFrame | None:
    logger.info("Cargando dataset desde {path}")
    #Cargar datos
    try:
        df = pd.read_csv('data/competencia_01.csv')
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset {e}")
        raise

def convertir_clase_ternaria_a_target(df:pd.DataFrame)->pd.DataFrame:
    # TODO: Implementar la función convertir_clase_ternaria_a_target
    logger.info("Convirtiendo clase_ternaria a target binario")

    df = df.rename(columns={"target": "clase_ternaria"})

    df_result = df.copy()

    # Contar valores originales para logging
    n_continua_orig = (df_result['clase_ternaria'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['clase_ternaria'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['clase_ternaria'] == 'BAJA+2').sum()

    df_result['clase_ternaria'] = df_result['clase_ternaria'].map({'CONTINUA':0,
                                                                  'BAJA+1':1,
                                                                  'BAJA+2':1})
    
    n_ceros = (df_result['clase_ternaria'] == 0).sum()
    n_unos = (df_result['clase_ternaria'] ==1).sum()

    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")
  
    return df_result

#ganancia_total = calcular_ganancia(y_val,y_pred_binary)


def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }

    

    # Crear la carpeta si no existe
    os.makedirs(os.path.dirname(archivo), exist_ok=True)
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")