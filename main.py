import pandas as pd
import CSVReader as csvReader
import json
from Assistant import Assistant

# Archivos a utilizar
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

# Variables de testing
id_carrera              = 206    # Ingenieria en Sistemas
id_plan                 = "2011" # Plan de Estudios Viejo

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


def combinar_archivos_para_gpt(archivo_combinar, archivo_alumnos, archivo_datos_hist_personales, archivo_plan_2011,
                               archivo_plan_2011_etiquetado, archivo_etiquetas, archivo_equivalencias, archivo_plan_2022):
    """
    Combinar un archivo de historia academica con demás archivos necesarios para el entrenamiento de chatGPT

    :param archivo_combinar: Archivo de historia academica a combinar
    :param archivo_plan_2022: Archivo con plan 2022
    :param archivo_equivalencias: Archivo con equivalencias del plan 2011
    :param archivo_etiquetas: Archivo con etiquetas para cada materia
    :param archivo_plan_2011_etiquetado: Archivo con el plan 2011 etiquetado
    :param archivo_plan_2011: Archivo con plan 2011
    :param archivo_datos_hist_personales: Archivo con datos personales historicos
    :param archivo_alumnos: Archivo con alumnos
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
    resultado_final = resultado_equivalencia.drop(['carrera_x', 'plan_x', 'resultado', 'comision', 'fin_vigencia_regul',
                                                   'nombre_materia_y', 'materia_y', 'cuatrimestre_y', 'anio',
                                                   'horas_teoria_y', 'horas_practica_y', 'id_persona', 'carrera_y', 'plan_y',
                                                   'fecha_inscripcion', 'regular', 'calidad', 'localidad_nacimiento',
                                                   'colegio_secundario', 'titulo_secundario', 'fecha_relevamiento',
                                                   'situacion_padre', 'situacion_madre', 'turno_preferido', 'es_celiaco',
                                                   'periodo_lectivo_localidad', 'periodo_lectivo_codigo_postal',
                                                   'periodo_lectivo_calle', 'periodo_lectivo_numero',
                                                   'procedencia_localidad', 'procedencia_codigo_postal', 'procedencia_calle',
                                                   'procedencia_numero', 'tipo_vivienda'], axis=1)

    return resultado_final

def evaluar_prediccion(alumno):
    """
    Predice la trayectoria académica de un alumno del plan 2011 en el plan 2022

    :param alumno: Identificador de alumno del cuál es quiere realizar la predicción
    """
    archivo_regularidades = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_regularidades,
                                                            id_carrera=id_carrera, id_plan=id_plan)
    archivo_alumnos = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_alumnos,
                                                      id_carrera=id_carrera, id_plan=id_plan)
    archivo_historia_academica = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_historia_academica,
                                                                 id_carrera=id_carrera, id_plan=id_plan)
    archivo_datos_hist_personales = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_personales)
    archivo_equivalencias = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_equivalencias)
    archivo_etiquetas = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_etiquetas)
    archivo_plan_2011 = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011)
    archivo_plan_2011_etiquetado = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011_etiquetado)
    archivo_plan_2022 = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2022)
    archivo_indice_exito_academico = open(ruta_archivo_indice_exito_academico, 'r').read()

    archivo_historia_academica['materia'] = archivo_historia_academica['materia'].astype(int)
    archivo_regularidades['materia'] = archivo_regularidades['materia'].astype(int)
    archivo_plan_2011['materia'] = archivo_plan_2011['materia'].astype(int)
    archivo_equivalencias['equivalencias_2022'] = archivo_equivalencias['equivalencias_2022'].astype(int)

    # Convierto fechas a datetime y me quedo solo con el anio
    archivo_datos_hist_personales['fecha_actualizacion'] = pd.to_datetime(
        archivo_datos_hist_personales['fecha_actualizacion'])
    archivo_datos_hist_personales['anio_actualizacion'] = archivo_datos_hist_personales['fecha_actualizacion'].dt.year
    archivo_datos_hist_personales = archivo_datos_hist_personales.drop('fecha_actualizacion', axis=1)
    # Elimino duplicados en datos historicos personales
    archivo_datos_hist_personales = archivo_datos_hist_personales.drop_duplicates(subset=['id_persona',
                                                                                          'anio_actualizacion'],
                                                                                  keep='first')

    archivo_historia_academica['fecha'] = pd.to_datetime(archivo_historia_academica['fecha'])
    archivo_historia_academica['anio_examen'] = archivo_historia_academica['fecha'].dt.year

    archivo_regularidades['fecha_regularidad'] = pd.to_datetime(archivo_regularidades['fecha_regularidad'])
    archivo_regularidades['anio_cursada'] = archivo_regularidades['fecha_regularidad'].dt.year

    # Filtro por un alumno particular, conservando el DataFrame original
    filtrado_alumno_finales = archivo_historia_academica[archivo_historia_academica['id_alumno'] == alumno]
    filtrado_alumno_finales = filtrado_alumno_finales.drop(['carrera',
                                                            'plan',
                                                            'fecha',
                                                            'resultado',
                                                            'forma_aprobacion'
                                                            ],
                                                           axis=1)
    filtrado_alumno_regularidades = archivo_regularidades[archivo_regularidades['id_alumno'] == alumno]
    ta_particular = combinar_archivos_para_gpt(filtrado_alumno_regularidades, archivo_alumnos,
                                               archivo_datos_hist_personales, archivo_plan_2011,
                                               archivo_plan_2011_etiquetado, archivo_etiquetas, archivo_equivalencias,
                                               archivo_plan_2022)

    # Obtengo los primeros X alumnos para entrenar al modelo con ellos
    primeros_alumnos = archivo_alumnos.head(10)
    # Me quedo sólo con sus IDs
    primeros_alumnos = primeros_alumnos['id_alumno'].tolist()

    # Filtro por un conjunto de alumnos, conservando el DataFrame original
    filtrado_primeros_alumnos = archivo_regularidades[
        archivo_regularidades['id_alumno'].isin(primeros_alumnos)]
    ta_entrenamiento = combinar_archivos_para_gpt(filtrado_primeros_alumnos, archivo_alumnos,
                                               archivo_datos_hist_personales, archivo_plan_2011,
                                               archivo_plan_2011_etiquetado, archivo_etiquetas, archivo_equivalencias,
                                               archivo_plan_2022)

    # Creación de instancia para la comunicación con chatGPT
    asistente = Assistant(
        archivo_indice_exito_academico=archivo_indice_exito_academico
    )

    # Obtengo predicción utilizando chatGPT
    datos_proyectados = asistente.procesar_archivo_con_gpt4(ta_particular, archivo_plan_2022, ta_entrenamiento)

    return datos_proyectados