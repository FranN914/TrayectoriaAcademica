import CSVReader as csvReader
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.losses import MeanSquaredError


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
    # Convierto las columnas con datos numéricos almacenados como texto
    if col == 'plan' or col == 'materia':
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    # Convierto las columnas de texto a columnas categoricas
    if col == 'cond_regularidad' or col == 'resultado' or col == 'calidad':
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

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





# Definimos el tamaño de entrada y latente
input_dim = X.shape[1]  # Número de características
latent_dim = 5          # Dimensión del espacio latente

# Codificador
inputs = layers.Input(shape=(input_dim,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Muestra del espacio latente
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Decodificador
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(32, activation='relu')(decoder_inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(input_dim, activation='sigmoid')(x)

# Construcción del modelo
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_inputs, outputs, name='decoder')
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# Supongamos que ya tienes tus variables 'inputs', 'outputs', 'z_mean', 'z_log_var' y 'input_dim' configuradas.

# Crear el modelo VAE (solo la parte de la arquitectura, no pérdidas)
inputs = layers.Input(shape=(input_dim,))
outputs = layers.Dense(input_dim, activation='sigmoid')(inputs)  # ejemplo de capa de salida

# Aquí defines la media y varianza en un VAE simple
z_mean = layers.Dense(latent_dim)(inputs)
z_log_var = layers.Dense(latent_dim)(inputs)

# Reconstrucción (Mean Squared Error)
def compute_reconstruction_loss(inputs, outputs):
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=-1)
    reconstruction_loss = tf.reduce_sum(reconstruction_loss) * input_dim
    return reconstruction_loss

# Divergencia KL
def compute_kl_loss(z_mean, z_log_var):
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return kl_loss

# Calcula la pérdida en el entrenamiento
reconstruction_loss = compute_reconstruction_loss(inputs, outputs)
kl_loss = compute_kl_loss(z_mean, z_log_var)

# Total VAE loss
vae_loss = reconstruction_loss + kl_loss

# Creando el modelo final
vae = Model(inputs, outputs)
vae.add_loss(vae_loss)

# Compilar el modelo
vae.compile(optimizer='adam')

vae.fit(X, X, epochs=50, batch_size=32)