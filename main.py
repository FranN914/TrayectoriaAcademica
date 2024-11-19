import CSVReader as csvReader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# # Identificar columnas de tipo object
# columnas_object = X_train.select_dtypes(include=['object']).columns
# print("Columnas categóricas o de texto:", columnas_object)
#
# for col in columnas_object:
#     print(f"Columna '{col}':")
#     print(X_train[col].unique())  # Valores únicos en la columna

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
    # Convierto las columnas con datos numéricos almacenados como texto
    if col == 'plan' or col == 'materia':
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    # Convierto las columnas de texto a columnas categoricas
    if col == 'cond_regularidad' or col == 'resultado' or col == 'calidad':
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])  # Usa el mismo mapeo para X_test

# # Asegurar consistencia en las columnas de X_train y X_test
# X_train, X_test = X_train.align(X_test, join='inner', axis=1)

# print(X_train.dtypes)  # Muestra los tipos de datos de cada columna

# Relleno valores faltantes (NaN) de X_train con 0
for col in X_train:
    X_train.loc[:, col] = X_train[col].fillna(0)

# Convierto las columnas con datos numéricos almacenados como texto
y_train = pd.to_numeric(y_train, errors='coerce')

# Relleno valores faltantes (NaN) de y_train con 0
y_train = y_train.fillna(0)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predecir
y_pred = modelo.predict(X_test)

# # Evaluar
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
