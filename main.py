import pandas as pd

import CSVReader as csvReader
from Assistant import Assistant
from openai import OpenAI
import json

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
    # Combino el resultado con las materias del plan 2011
    resultado = pd.merge(archivo_combinar, archivo_plan_2011, on='materia', how='inner')

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

    # Selecciono que columnas quiero ver
    resultado_final = resultado_equivalencia.drop(['carrera', 'plan', 'resultado', 'forma_aprobacion', 'nombre_materia_y',
                                                   'materia_y', 'cuatrimestre_y', 'anio', 'horas_teoria_y',
                                                   'horas_practica_y'], axis=1)

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
# archivo_datos_hist_personales       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_personales)
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

# Comienza la implementación

archivo_historia_academica['materia'] = archivo_historia_academica['materia'].astype(int)
archivo_plan_2011['materia'] = archivo_plan_2011['materia'].astype(int)
archivo_equivalencias['equivalencias_2022'] = archivo_equivalencias['equivalencias_2022'].astype(int)

# Filtro por un alumno particular, conservando el DataFrame original
filtrado_alumno = archivo_historia_academica[archivo_historia_academica['id_alumno'] == id_alumno]

ha_particular = combinar_archivos_para_gpt(filtrado_alumno)

# Muestro solo las columnas que quiero
# print(resultado_equivalencia[columnas_visibles])

# Comienzo de preparación de datos para entrenamiento

# Obtengo los primeros 50 alumnos para entrenar al modelo con ellos
primeros_alumnos = archivo_alumnos.head(20)
# Me quedo sólo con sus IDs
primeros_alumnos = primeros_alumnos['id_alumno'].tolist()

# Filtro por un conjunto de alumnos, conservando el DataFrame original
filtrado_primeros_alumnos = archivo_historia_academica[archivo_historia_academica['id_alumno'].isin(primeros_alumnos)]

ha_entrenamiento = combinar_archivos_para_gpt(filtrado_primeros_alumnos)

# asistente = Assistant(
#     archivo_plan_2011='',
#     archivo_plan_2011_etiquetado='',
#     archivo_plan_2011_precedencia='',
#     archivo_plan_2022='',
#     archivo_plan_2022_precedencia='',
#     archivo_equivalencias='',
#     archivo_etiquetas='',
#     archivo_historia_academica=archivo_historia_academica,
#     archivo_indice_exito_academico=''
#     # archivo_equivalencias=archivo_equivalencias,
#     # archivo_etiquetas=archivo_etiquetas,
#     # archivo_historia_academica=archivo_historia_academica,
#     # archivo_indice_exito_academico=archivo_indice_exito_academico
# )
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