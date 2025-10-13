import pandas as pd
import os
import datetime
import logging
import optuna 
from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag,feature_engineering_rolling_avg
from src.conf import *
from src.optimization import optimizar,evaluar_en_test
from src.best_params import cargar_mejores_hiperparametros
from src.final_train import generar_predicciones_finales,entrenar_modelo_final,preparar_datos_entrenamiento_final,guardar_modelo,guardar_predicciones_finales

#config basico logging
os.makedirs('logs',exist_ok=True)



fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [
        logging.FileHandler(f'logs/{nombre_log}',mode='w',encoding='utf-8'),
        logging.StreamHandler()
    ]
    )

logger = logging.getLogger("__name__")

# MANEJO DE CONFIGURACION EN YAML
logger.info("CONFIGURACION CARGADA DESDE YAML")
logger.info(f"Study_name:{STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"SEMILLA:{SEMILLA}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")
            


def main():
    logger.info("Inicio de ejecución.")

    os.makedirs("data",exist_ok=True)
    PATH = DATA_PATH
    df= cargar_datos(PATH)

    #Feature Engineering
    atributos = ['mrentabilidad','mrentabilidad_annual','mcomisiones','mactivos_margen','mpasivos_margen',
        'cproductos','tcuentas','mcuenta_corriente_adicional','mcuenta_corriente','ccaja_ahorro',
        'mcaja_ahorro','mcaja_ahorro_adicional','mcaja_ahorro_dolares','mcuentas_saldo','ctarjeta_debito',
        'ctarjeta_debito_transacciones','mautoservicio','ctarjeta_visa','mtarjeta_visa_consumo','ctarjeta_master',
        'ctarjeta_master_transacciones','mtarjeta_master_consumo','cprestamos_personales','mprestamos_personales','cprestamos_hipotecarios',
        'mprestamos_hipotecarios','cplazo_fijo','mplazo_fijo_dolares','mplazo_fijo_pesos','cinversion1',
        'minversion1_pesos','minversion1_dolares','cinversion2','minversion2','cpayroll_trx',
        'mpayroll','mpayroll2','cpayroll2_trx','ccuenta_debitos_automaticos','mcuenta_debitos_automaticos',
        'ctarjeta_visa_debitos_automaticos','mttarjeta_visa_debitos_automaticos','ctarjeta_master_debitos_automaticos','mttarjeta_master_debitos_automaticos','cpagodeservicios',
        'mpagodeservicios','cpagomiscuentas','mpagomiscuentas','ccajeros_propios_descuentos','mcajeros_propios_descuentos',
        'ctarjeta_visa_descuentos','mtarjeta_visa_descuentos','mtarjeta_master_descuentos','mcomisiones_mantenimiento','ccomisiones_otras',
        'mcomisiones_otras','cforex','cforex_buy','mforex_buy','cforex_sell',
        'mforex_sell','ctransferencias_recibidas','mtransferencias_recibidas','ctransferencias_emitidas','mtransferencias_emitidas',
        'cextraccion_autoservicio','mextraccion_autoservicio','ccheques_depositados','mcheques_depositados','ccheques_emitidos',
        'mcheques_emitidos','ccheques_depositados_rechazados','mcheques_depositados_rechazados','ccheques_emitidos_rechazados','mcheques_emitidos_rechazados',
        'ccallcenter_transacciones','chomebanking_transacciones','ccajas_transacciones','ccajas_consultas','ccajas_depositos',
        'ccajas_extracciones','ccajas_otras','catm_trx','matm','catm_trx_other',
        'matm_other','ctrx_quarter','cmobile_app_trx','Master_mfinanciacion_limite','Master_msaldototal',
        'Master_msaldopesos','Master_msaldodolares','Master_mconsumospesos','Master_mconsumosdolares','Master_mlimitecompra',
        'Master_madelantopesos','Master_madelantodolares','Master_mpagado','Master_mpagospesos','Master_mpagosdolares',
        'Master_mconsumototal','Master_cconsumos','Master_cadelantosefectivo','Master_mpagominimo','Visa_mfinanciacion_limite',
        'Visa_msaldototal','Visa_msaldopesos','Visa_msaldodolares','Visa_mconsumospesos','Visa_mconsumosdolares',
        'Visa_mlimitecompra','Visa_madelantopesos','Visa_madelantodolares','Visa_mpagado','Visa_mpagospesos',
        'Visa_mpagosdolares','Visa_mconsumototal','Visa_cconsumos','Visa_mpagominimo'
    ]
    cant_lag =2
    df_fe = feature_engineering_lag(df,columnas=atributos,cant_lag=cant_lag)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    #calcula el AVG
    df_fe = feature_engineering_rolling_avg(df_fe, columnas=atributos)
    logger.info(f"Paso 2 (Rolling AVG 4M) completado: {df_fe.shape}") 
    
    #2 Convertir clase_ternaria a target binario
    df_fe = convertir_clase_ternaria_a_target(df_fe)
    logger.info(f"Convertir clase_ternaria a target binario completado: {df_fe.shape}")

    #03 Ejecutar optimización de hiperparametros
    study = optuna.create_study(study_name=STUDY_NAME,direction="maximize")
    study = optimizar(df_fe,n_trial=100)

  
    # 4. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    logger.info("Mejores hiperparametros: {study.best_params}")
    logger.info("Test en mes de test")
    mejores_params = cargar_mejores_hiperparametros()
    resultado_test = evaluar_en_test(df_fe,mejores_params)
    ganancia_test = resultado_test['ganancia_test']
    logger.info(f'Ganancia en mes de test:{ganancia_test:,.0f}')


    os.makedirs("data",exist_ok=True)
    path = "data/competencia_01_lag.csv"
    df.to_csv(path,index=False)
    
    #Entrenar modelo final
    #05 Entrenar modelo final
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
    modelo_final = entrenar_modelo_final(X_train, y_train, mejores_params)

    #Guardar modelo final
    guardar_modelo(modelo_final,FINAL_TRAIN)

    #06 Generar predicciones finales
    predicciones = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)

    #07 Guardar predicciones
    salida_kaggle = guardar_predicciones_finales(predicciones)

    

    logger.info(f">>> Ejecución finalizada. Revisar los detalles de log: {nombre_log} ")

if __name__ == '__main__':
    main()