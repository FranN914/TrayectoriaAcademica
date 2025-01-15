import pandas as pd
import CSVReader as csvReader
import json
from Assistant import Assistant

def formatear_json(input_json):
    """
    Formatea un JSON correctamente.

    :param input_json: Archivo JSON de entrada (ruta).
    """
    try:
        # Cargar el JSON
        with open(input_json, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Formatear contenido
        for entry in data:
            if "resultado" in entry and entry["resultado"].startswith("```json"):
                # Limpiar el bloque de código encapsulado
                json_string = entry["resultado"].strip("```json\n").strip("```")
                entry["resultado"] = json.loads(json_string)  # Convertir el string JSON en objeto

        # Guardar el JSON formateado
        with open(input_json, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"JSON formateado guardado en: {input_json}")

    except Exception as e:
        print(f"Error al procesar el archivo JSON: {e}")


def combinar_archivos_para_gpt(archivo_combinar):
    """
    Combinar un archivo de istoria academica con demás archivos necesarios para el entrenamiento de chatGPT

    :param archivo_combinar: Archivo de historia academica a combinar
    """
    # Combino el resultado con los datos de alumno para obtener el id_persona
    resultado_con_alumno = pd.merge(archivo_combinar, archivo_alumnos, on='id_alumno')

    # Combino el resultado con los datos personales
    resultado_datos_personales = pd.merge(resultado_con_alumno, archivo_datos_hist_personales, left_on=['id_persona', 'anio_cursada'], right_on=['id_persona', 'anio_actualizacion'], how='left')

    # Combino el resultado con las materias del plan 2011
    resultado = pd.merge(resultado_datos_personales, archivo_plan_2011, on='materia', how='inner')

    # Combino el resultado con las etiquetas de las materias del plan 2011
    resultado_2011_etiquetado = pd.merge(resultado, archivo_plan_2011_etiquetado, on='materia', how='inner')
    resultado_2011_etiquetado = pd.merge(resultado_2011_etiquetado, archivo_etiquetas, on='id_etiqueta', how='inner')

    # Combino el resultado con las equivalencias de las materias del plan 2011
    resultado_equivalencia = pd.merge(resultado_2011_etiquetado, archivo_equivalencias, on='materia', how='left')

    # Relleno valores faltantes con 0
    resultado_equivalencia['equivalencias_2022'] = resultado_equivalencia['equivalencias_2022'].fillna(0)
    resultado_equivalencia['equivalencias_2022'] = resultado_equivalencia['equivalencias_2022'].astype(int)

    # Combino el resultado con las materias del plan 2022 para obtener su nombre
    resultado_equivalencia = pd.merge(resultado_equivalencia, archivo_plan_2022, left_on='equivalencias_2022',
                                      right_on='materia', how='left')

    # Reemplazo valores NaN con 0 para columna anio actualizacion, y la interpreto como int
    resultado_equivalencia['anio_actualizacion'] = resultado_equivalencia['anio_actualizacion'].fillna(0).astype(int)

    # Selecciono que columnas quiero ver
    resultado_final = resultado_equivalencia.drop(['carrera_x', 'plan_x', 'resultado', 'forma_aprobacion', 'nombre_materia_y',
                                                   'materia_y', 'cuatrimestre_y', 'anio', 'horas_teoria_y',
                                                   'horas_practica_y', 'id_persona', 'carrera_y', 'plan_y',
                                                   'fecha_inscripcion', 'regular', 'calidad', 'localidad_nacimiento',
                                                   'colegio_secundario', 'titulo_secundario', 'fecha_relevamiento',
                                                   'situacion_padre', 'situacion_madre', 'turno_preferido', 'es_celiaco',
                                                   'periodo_lectivo_localidad', 'periodo_lectivo_codigo_postal',
                                                   'periodo_lectivo_calle', 'periodo_lectivo_numero',
                                                   'procedencia_localidad', 'procedencia_codigo_postal', 'procedencia_calle',
                                                   'procedencia_numero', 'tipo_vivienda'], axis=1)

    return resultado_final

### Archivos a utilizar
ruta_archivo_regularidades              = f"DataSource/002_regularidades.csv"
ruta_archivo_alumnos                    = f"DataSource/001_alumnos.csv"
ruta_archivo_historia_academica         = f"DataSource/003_historia_Academica.csv"
ruta_archivo_datos_personales           = f"DataSource/101_datos_personales.csv"
ruta_archivo_datos_laborales            = f"DataSource/103_financimiento_y_datos_laborales.csv"
ruta_archivo_datos_hist_personales      = f"DataSource/201_hist_datos_personales.csv"
ruta_archivo_datos_hist_laborales       = f"DataSource/203_hist_financimiento_y_datos_laborales.csv"
ruta_archivo_equivalencias              = f"DataSource/equivalencias.csv"
ruta_archivo_etiquetas                  = f"DataSource/etiquetas.csv"
ruta_archivo_optativas_etiquetado       = f"DataSource/optativas_etiquetado.csv"
ruta_archivo_plan_2011                  = f"DataSource/plan_2011.csv"
ruta_archivo_plan_2011_etiquetado       = f"DataSource/plan_2011_etiquetado.csv"
ruta_archivo_plan_2011_precedencia      = f"DataSource/plan_2011_precedencia.csv"
ruta_archivo_plan_2022                  = f"DataSource/plan_2022.csv"
ruta_archivo_plan_2022_precedencia      = f"DataSource/plan_2022_precedencia.csv"
ruta_archivo_indice_exito_academico     = f"DataSource/indice_exito_academico.txt"

### Variables de testing
id_carrera              = 206    # Ingenieria en Sistemas
id_plan                 = "2011" # Plan de Estudios Viejo
id_plan_nuevo           = "2022" # Plan de Estudios Nuevo
id_alumno               = 60451  # Alumno
equivalencias           = True   # Variable para determinar si trabajar con equivalencias o no

# Apertura de archivos
# archivo_regularidades               = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_regularidades, id_carrera=id_carrera, id_plan=id_plan_nuevo)
archivo_alumnos                     = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_alumnos, id_carrera=id_carrera, id_plan=id_plan)
archivo_historia_academica          = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_historia_academica, id_carrera=id_carrera, id_plan=id_plan)
# archivo_datos_personales            = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_personales)
# archivo_datos_laborales             = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_laborales)
archivo_datos_hist_personales       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_personales)
# archivo_datos_hist_laborales        = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_laborales)
archivo_equivalencias               = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_equivalencias)
archivo_etiquetas                   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_etiquetas)
# archivo_optativas_etiquetado        = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_optativas_etiquetado)
archivo_plan_2011                   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011)
archivo_plan_2011_etiquetado        = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011_etiquetado)
# archivo_plan_2011_precedencia       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011_precedencia)
archivo_plan_2022                   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2022)
# archivo_plan_2022_precedencia       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2022_precedencia)
archivo_indice_exito_academico      = open(ruta_archivo_indice_exito_academico, 'r').read()

''' Implementación del modelo '''

# Comienzo de preparación de datos para entrenamiento

archivo_historia_academica['materia'] = archivo_historia_academica['materia'].astype(int)
archivo_plan_2011['materia'] = archivo_plan_2011['materia'].astype(int)
archivo_equivalencias['equivalencias_2022'] = archivo_equivalencias['equivalencias_2022'].astype(int)

# Convierto fechas a datetime y me quedo solo con el anio
archivo_datos_hist_personales['fecha_actualizacion'] = pd.to_datetime(archivo_datos_hist_personales['fecha_actualizacion'])
archivo_datos_hist_personales['anio_actualizacion'] = archivo_datos_hist_personales['fecha_actualizacion'].dt.year
archivo_datos_hist_personales = archivo_datos_hist_personales.drop('fecha_actualizacion', axis=1)   # Elimino la columna, ya no la usamos mas
archivo_historia_academica['fecha'] = pd.to_datetime(archivo_historia_academica['fecha'])
archivo_historia_academica['anio_cursada'] = archivo_historia_academica['fecha'].dt.year

# Elimino duplicados en datos historicos personales
archivo_datos_hist_personales = archivo_datos_hist_personales.drop_duplicates(subset=['id_persona', 'anio_actualizacion'], keep='first')

# Filtro por un alumno particular, conservando el DataFrame original
filtrado_alumno = archivo_historia_academica[archivo_historia_academica['id_alumno'] == id_alumno]
ha_particular = combinar_archivos_para_gpt(filtrado_alumno)

# Obtengo los primeros 50 alumnos para entrenar al modelo con ellos
primeros_alumnos = archivo_alumnos.head(20)
# Me quedo sólo con sus IDs
primeros_alumnos = primeros_alumnos['id_alumno'].tolist()

# Filtro por un conjunto de alumnos, conservando el DataFrame original
# filtrado_primeros_alumnos = archivo_historia_academica[archivo_historia_academica['id_alumno'].isin(primeros_alumnos)]
# ha_entrenamiento = combinar_archivos_para_gpt(filtrado_primeros_alumnos)

# Fin de preparación de datos para entrenamiento

# Comienza implementación del modelo

# # Creación de instancia para la comunicación con chatGPT
# asistente = Assistant(
#     archivo_indice_exito_academico=''
# )
#
# # Obtengo predicción utilizando chatGPT
# datos_proyectados = asistente.procesar_archivo_con_gpt4(ha_particular, archivo_plan_2022, ha_entrenamiento)
#
# print("Datos Proyectados:", datos_proyectados)

#
# # Fin de la implementación
#
# # Nombre del archivo CSV de salida
# archivo_json = "resultados.json"
#
# # Procesar los archivos y generar el CSV
# asistente.procesar_archivos_y_generar_json(archivo_json, id_alumno, archivo_json)
#
# formatear_json(archivo_json)

# #Comienzo del entrenamiento

# #Fin del entrenamiento

# # Comienzo de la visualización de resultados
# print(chat_completion.choices[0].message)