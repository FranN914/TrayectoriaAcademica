import CSVReader as csvReader
import keras
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler

### Archivos a utilizar
ruta_archivo_regularidades              = f"DataSource/002_regularidades.csv"
ruta_archivo_materias                   = f"DataSource/000_materias_planes.csv"
ruta_archivo_alumnos                    = f"DataSource/001_alumnos.csv"
ruta_archivo_datos_personales           = f"DataSource/101_datos_personales.csv"
ruta_archivo_datos_laborales            = f"DataSource/103_financimiento_y_datos_laborales.csv"
ruta_archivo_datos_actividades          = f"DataSource/104_datos_actividades.csv"
ruta_archivo_datos_docente              = f"DataSource/106_actuacion_docente.csv"
ruta_archivo_datos_discapacidad         = f"DataSource/107_discapacidad.csv"
ruta_archivo_datos_hist_personales      = f"DataSource/201_hist_datos_personales.csv"
ruta_archivo_datos_hist_laborales       = f"DataSource/203_hist_financimiento_y_datos_laborales.csv"
ruta_archivo_datos_hist_actividades     = f"DataSource/204_hist_datos_actividades.csv"
ruta_archivo_datos_hist_discapacidad    = f"DataSource/207_hist_discapacidad.csv"

### Variables de testing
id_carrera              = 206    # Ingenieria en Sistemas
id_plan                 = "2011" # Plan de Estudios Viejo
id_plan_equivalencias   = "2022" # Plan de Estudios Nuevo
id_persona              = 54294  # Persona: Benito Juarez
equivalencias           = True   # Variable para determinar si trabajar con equivalencias o no

# Apertura de archivos
archivo_materias_plan_viejo     = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_materias, id_carrera=id_carrera, id_plan=id_plan)
archivo_materias_plan_nuevo     = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_materias, id_carrera=id_carrera, id_plan=id_plan_equivalencias)
archivo_regularidades           = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_regularidades, id_carrera=id_carrera, id_plan=id_plan_equivalencias)
archivo_alumnos                 = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_alumnos, id_carrera=id_carrera, id_plan=id_plan)
archivo_datos_personales        = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_personales, id_persona=id_persona)
archivo_datos_laborales         = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_laborales, id_persona=id_persona)
archivo_datos_actividades       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_actividades, id_persona=id_persona)
archivo_datos_docente           = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_docente, id_persona=id_persona)
archivo_datos_discapacidad      = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_discapacidad, id_persona=id_persona)
archivo_datos_hist_personales   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_personales, id_persona=id_persona)
archivo_datos_hist_laborales    = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_laborales, id_persona=id_persona)
archivo_datos_hist_actividades  = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_actividades, id_persona=id_persona)
archivo_datos_hist_discapacidad = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_discapacidad, id_persona=id_persona)


''' Implementación del modelo '''

# Combinar los alumnos con las materias cursadas y su regularidad
datos_entrenamiento = pd.merge(archivo_regularidades, archivo_alumnos, on=['id_alumno', 'carrera', 'plan'], how='outer')

# Separar características (X) y etiqueta (y)
X = datos_entrenamiento[['id_alumno', 'id_persona', 'carrera', 'plan', 'materia', 'fecha_regularidad', 'cond_regularidad', 'resultado', 'calidad']]
y = datos_entrenamiento['nota']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convierto las columnas con fechas a columnas numericas
X_train['anio_regularidad'] = pd.to_datetime(X_train['fecha_regularidad']).dt.year
X_train['mes_regularidad'] = pd.to_datetime(X_train['fecha_regularidad']).dt.month
X_train['día_regularidad'] = pd.to_datetime(X_train['fecha_regularidad']).dt.day
X_train = X_train.drop(columns=['fecha_regularidad'])  # Opcional, elimina la columna original
# Lo mismo para X_test
X_test['anio_regularidad'] = pd.to_datetime(X_test['fecha_regularidad']).dt.year
X_test['mes_regularidad'] = pd.to_datetime(X_test['fecha_regularidad']).dt.month
X_test['día_regularidad'] = pd.to_datetime(X_test['fecha_regularidad']).dt.day
X_test = X_test.drop(columns=['fecha_regularidad'])


le = LabelEncoder()
for col in X_train:
    # Convierto las columnas de texto a columnas categoricas
    if col == 'cond_regularidad' or col == 'resultado' or col == 'calidad':
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    else:
        # Convierto las columnas con datos numéricos almacenados como texto
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(int)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(int)

# Relleno valores faltantes (NaN) de X_train con 0
for col in X_train:
    X_train.loc[:, col] = X_train[col].fillna(0)
    X_test.loc[:, col] = X_test[col].fillna(0)

# Convierto las columnas con datos numéricos almacenados como texto
y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')

# Relleno valores faltantes (NaN) de y_train con 0
y_train = y_train.fillna(0)
y_test = y_test.fillna(0)

"""
# Normalización de los datos
"""

# Inicializa el escalador
scaler = MinMaxScaler()

# Aplica normalización a X_train y X_test
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Si y_train también necesita normalización
y_train_normalized = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_normalized = scaler.transform(y_test.values.reshape(-1, 1))

"""
# Fin de normalización de los datos
"""

"""
# Sección para ver datos de entenamiento

# Convierte a DataFrame para inspección si es necesario
X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns)
y_train_normalized_df = pd.DataFrame(y_train_normalized, columns=["target"])

for col in X_train.columns:
    original_min = X_train[col].min()
    original_max = X_train[col].max()
    normalized_min = X_train_normalized_df[col].min()
    normalized_max = X_train_normalized_df[col].max()

    print(f"Columna: {col}")
    print(f"  Mínimo original: {original_min}, Mínimo normalizado: {normalized_min}")
    print(f"  Máximo original: {original_max}, Máximo normalizado: {normalized_max}")
    print("-" * 40)
"""

"""
#Comienza la implementación
"""

"""
#Fin de la implementación 
"""

"""
#Comienza el entrenamiento
"""