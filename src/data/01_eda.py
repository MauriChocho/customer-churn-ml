import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/customer_churn_historical.csv")


def run_eda(filepath: Path = DATA_PATH):
    """Análisis exploratorio del dataset de churn de clientes."""
    if not filepath.exists():
        print(f"Error: no se encontró el archivo '{filepath}'.")
        print("Correr 'dvc pull' para bajar los datos desde DagsHub.")
        return

    df = pd.read_csv(filepath)

    #  Dimensiones y tipos de datos 
    print("   Dimensiones del dataset   ")
    print(f"Filas: {df.shape[0]}, columnas: {df.shape[1]}\n")
    print("**************************\n")
    
    print("   Tipos de datos   ")
    df.info()
    print()
    print("**************************\n")
    #     Calidad de datos    
    print("    Duplicados    ")
    duplicados = df.duplicated().sum()
    print(f"Filas duplicadas: {duplicados}\n")
    print("**************************\n")

    print("    Validación de customerID    ")
    if "customerID" in df.columns:
        clientes_unicos = df["customerID"].nunique()
        total_registros = len(df)
        ids_nulos = df["customerID"].isna().sum()

        print(f"Clientes únicos: {clientes_unicos}")
        print(f"Total de registros: {total_registros}")
        print(f"customerID nulos: {ids_nulos}")
        if clientes_unicos == total_registros and ids_nulos == 0:
            print("Validación OK: todos los customerID son únicos.\n")
        else:
            print("Validación FALLIDA: hay customerID duplicados o nulos.\n")
    else:
        print("No se encontró la columna 'customerID'.\n")

    print("**************************\n")
    print("   variable SeniorCitizen   ")
    if "SeniorCitizen" in df.columns:
        print(df["SeniorCitizen"].value_counts().sort_index())
        print("Se tratará como variable binaria con valores 0 y 1.\n")
    else:
        print("No se encontró la columna 'SeniorCitizen'.\n")

    print("    Valores nulos    ")
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print(nulos[nulos > 0])
    else:
        print("No hay valores nulos explícitos (NaN).")

    print("**************************\n")

    # TotalCharges viene como texto en el dataset original y tiene registros
    # con espacios en blanco en vez de números, hay que revisarlo aparte
    total_charges_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    espacios_en_blanco = total_charges_num.isnull().sum() - df["TotalCharges"].isnull().sum()
    print(f"Registros con formato inválido en TotalCharges: {espacios_en_blanco}\n")
    print("**************************\n")

    #     Distribución del target    
    print("    Distribución de Churn    ")
    if "Churn" in df.columns:
        distribucion = df["Churn"].value_counts(normalize=True).round(4) * 100
        print(distribucion)
        print("\nHay desbalance entre clases, la clase minoritaria representa "
              f"aproximadamente {distribucion.min():.1f}% del total.\n")
    else:
        print("No se encontró la columna 'Churn'.\n")
    print("**************************\n")

    #     Estadísticas descriptivas    
    print("    Estadísticas descriptivas (variables numéricas)    ")
    df_num = df.copy()
    df_num["TotalCharges"] = total_charges_num
    print(df_num.describe().round(2))
    print("**************************\n")

    print("    Variables categóricas: valores únicos    ")
    categoricas = df.select_dtypes(include="object").columns.drop(["customerID", "Churn"], errors="ignore")
    for col in categoricas:
        print(f"{col}: {df[col].nunique()} valores distintos -> {df[col].unique()}")

    print("**************************\n")

    #     Relación de variables con el target    
    if "Churn" in df.columns:
        print("    Promedio de variables numéricas según Churn    ")
        print(df_num.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean().round(2))
        print("**************************\n")

        print("    Proporción de Churn según tipo de contrato    ")
        print((pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100).round(2))


if __name__ == "__main__":
    run_eda()