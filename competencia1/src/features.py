import pandas as pd
import duckdb
import logging

logger = logging.getLogger("__name__")

def feature_engineering_lag(df:pd.DataFrame,columnas:list[str],cant_lag: int =1) ->pd.DataFrame:
    """
    Genera variables de lag para los atributos específicados utilizando sql
    
    Parameters:
    --------------
    df : pd.DataFrame
         DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df 
    
    sql = "SELECT *"

    #Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for lag in range(1, cant_lag + 1):
                sql += f", LAG({attr}, {lag}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag{lag}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql +=" FROM df"
    print(sql)
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df",df) 
    df = con.execute(sql).df()
    con.close()

    

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")
    
    return df




def feature_engineering_rolling_avg(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de promedio móvil (Rolling Average, 4m) para las columnas 
    especificadas utilizando SQL (DuckDB).
    """

    logger.info(f"Realizando Feature Engineering de Rolling Average (ventana 4m) para {len(columnas)} atributos")

    if not columnas:
        return df 
    
    sql_features_list = []

    for attr in columnas:
        # Generar ROLLING AVERAGE (Ventana de 4: 3 Precedentes + Actual)
        # ⚠️ Esto solo mira la columna {attr} a lo largo del tiempo.
        avg_sql = f"AVG({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS avg_4m_{attr}"
        sql_features_list.append(avg_sql)


    sql = "SELECT *, " + ", ".join(sql_features_list) + " FROM df"
    
    # Ejecutar la consulta con DuckDB
    con = duckdb.connect(database=":memory:")
    con.register("df", df) 
    df_result = con.execute(sql).df()
    con.close()

    logger.info("Rolling Average Feature Engineering completado.")
    
    return df_result

