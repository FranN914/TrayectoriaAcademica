import csv
from io import StringIO
import json
from typing import Optional
import pandas as pd
from pydantic import BaseModel
import CSVReader as csvReader
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

# Constantes
id_carrera  = 206    # Ingenieria en Sistemas
id_plan     = "2011" # Plan de Estudios Viejo

def combinar_archivos_para_gpt(archivo_combinar, archivo_alumnos, archivo_datos_hist_personales, archivo_plan_2011,
                               archivo_plan_2011_etiquetado, archivo_etiquetas, archivo_equivalencias,
                               archivo_plan_2022):
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
    resultado_con_alumno = pd.merge(archivo_combinar,
                                    archivo_alumnos,
                                    on='id_alumno')

    # Combino el resultado con los datos personales
    resultado_datos_personales = pd.merge(resultado_con_alumno,
                                          archivo_datos_hist_personales,
                                          left_on=['id_persona', 'anio_cursada'],
                                          right_on=['id_persona', 'anio_actualizacion'],
                                          how='left')

    # Combino el resultado con las materias del plan 2011
    resultado = pd.merge(resultado_datos_personales,
                         archivo_plan_2011,
                         on='materia',
                         how='inner')

    # Combino el resultado con las etiquetas de las materias del plan 2011
    resultado_2011_etiquetado = pd.merge(resultado,
                                         archivo_plan_2011_etiquetado,
                                         on='materia',
                                         how='inner')
    resultado_2011_etiquetado = pd.merge(resultado_2011_etiquetado,
                                         archivo_etiquetas,
                                         on='id_etiqueta',
                                         how='inner')

    # Combino el resultado con las equivalencias de las materias del plan 2011
    resultado_equivalencia = pd.merge(resultado_2011_etiquetado,
                                      archivo_equivalencias,
                                      on='materia',
                                      how='left')

    # Relleno valores faltantes con 0
    resultado_equivalencia['equivalencias_2022'] = resultado_equivalencia['equivalencias_2022'].fillna(0)
    resultado_equivalencia['equivalencias_2022'] = resultado_equivalencia['equivalencias_2022'].astype(int)

    # Combino el resultado con las materias del plan 2022 para obtener su nombre
    resultado_equivalencia = pd.merge(resultado_equivalencia,
                                      archivo_plan_2022,
                                      left_on='equivalencias_2022',
                                      right_on='materia',
                                      how='left')

    # Reemplazo valores NaN con 0 para columna anio actualizacion, y la interpreto como int
    resultado_equivalencia['anio_actualizacion'] = resultado_equivalencia['anio_actualizacion'].fillna(0).astype(int)

    # Selecciono que columnas quiero ver
    resultado_final = resultado_equivalencia.drop(['carrera_x',
                                                   'plan_x',
                                                   'resultado',
                                                   'comision',
                                                   'fin_vigencia_regul',
                                                   'nombre_materia_y',
                                                   'materia_y',
                                                   'cuatrimestre_y',
                                                   'anio',
                                                   'horas_teoria_y',
                                                   'horas_practica_y',
                                                   'id_persona',
                                                   'carrera_y',
                                                   'plan_y',
                                                   'fecha_inscripcion',
                                                   'regular',
                                                   'calidad',
                                                   'localidad_nacimiento',
                                                   'colegio_secundario',
                                                   'titulo_secundario',
                                                   'fecha_relevamiento',
                                                   'situacion_padre',
                                                   'situacion_madre',
                                                   'turno_preferido',
                                                   'es_celiaco',
                                                   'periodo_lectivo_localidad',
                                                   'periodo_lectivo_codigo_postal',
                                                   'periodo_lectivo_calle',
                                                   'periodo_lectivo_numero',
                                                   'procedencia_localidad',
                                                   'procedencia_codigo_postal',
                                                   'procedencia_calle',
                                                   'procedencia_numero',
                                                   'tipo_vivienda'],
                                                  axis=1)

    return resultado_final

def evaluar_prediccion(alumno):
    """
    Predice la trayectoria académica de un alumno del plan 2011 en el plan 2022

    :param alumno: Identificador de alumno del cuál es quiere realizar la predicción
    """
    archivo_alumnos                 = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_alumnos,
                                                                      id_carrera=id_carrera,
                                                                      id_plan=id_plan)
    archivo_regularidades           = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_regularidades,
                                                                      id_carrera=id_carrera,
                                                                      id_plan=id_plan)
    archivo_historia_academica      = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_historia_academica,
                                                                      id_carrera=id_carrera,
                                                                      id_plan=id_plan)
    archivo_datos_hist_personales   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_personales)
    archivo_equivalencias           = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_equivalencias)
    archivo_etiquetas               = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_etiquetas)
    archivo_plan_2011               = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011)
    archivo_plan_2011_etiquetado    = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011_etiquetado)
    archivo_plan_2022               = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2022)
    archivo_indice_exito_academico  = open(ruta_archivo_indice_exito_academico, 'r').read()

    archivo_historia_academica['materia']       = archivo_historia_academica['materia'].astype(int)
    archivo_regularidades['materia']            = archivo_regularidades['materia'].astype(int)
    archivo_plan_2011['materia']                = archivo_plan_2011['materia'].astype(int)
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

    archivo_historia_academica['fecha']         = pd.to_datetime(archivo_historia_academica['fecha'])
    archivo_historia_academica['anio_examen']   = archivo_historia_academica['fecha'].dt.year

    archivo_regularidades['fecha_regularidad']  = pd.to_datetime(archivo_regularidades['fecha_regularidad'])
    archivo_regularidades['anio_cursada']       = archivo_regularidades['fecha_regularidad'].dt.year

    # Filtro por un alumno particular, conservando el DataFrame original
    filtrado_alumno_finales = archivo_historia_academica[archivo_historia_academica['id_alumno'] == alumno]
    if filtrado_alumno_finales.empty:  # True si el DataFrame NO está vacío
        return {"resultado": "alumno inexistente"}

    filtrado_alumno_finales = filtrado_alumno_finales.drop(['carrera',
                                                            'plan',
                                                            'fecha',
                                                            'resultado',
                                                            'forma_aprobacion'
                                                            ],
                                                           axis=1)
    filtrado_alumno_regularidades = archivo_regularidades[archivo_regularidades['id_alumno'] == alumno]
    ta_particular = combinar_archivos_para_gpt(filtrado_alumno_regularidades,
                                               archivo_alumnos,
                                               archivo_datos_hist_personales,
                                               archivo_plan_2011,
                                               archivo_plan_2011_etiquetado,
                                               archivo_etiquetas,
                                               archivo_equivalencias,
                                               archivo_plan_2022)

    # Obtengo los primeros X alumnos para entrenar al modelo con ellos
    primeros_alumnos = archivo_alumnos.head(10)
    # Me quedo sólo con sus IDs
    primeros_alumnos = primeros_alumnos['id_alumno'].tolist()

    # Filtro por un conjunto de alumnos, conservando el DataFrame original
    filtrado_primeros_alumnos = archivo_regularidades[
        archivo_regularidades['id_alumno'].isin(primeros_alumnos)]
    ta_entrenamiento = combinar_archivos_para_gpt(filtrado_primeros_alumnos,
                                                  archivo_alumnos,
                                                  archivo_datos_hist_personales,
                                                  archivo_plan_2011,
                                                  archivo_plan_2011_etiquetado,
                                                  archivo_etiquetas,
                                                  archivo_equivalencias,
                                                  archivo_plan_2022)

    # Creación de instancia para la comunicación con chatGPT
    asistente = Assistant(archivo_indice_exito_academico=archivo_indice_exito_academico)

    # Obtengo predicción utilizando chatGPT
    datos_proyectados = asistente.procesar_archivo_con_gpt4(ta_particular,
                                                            archivo_plan_2022,
                                                            ta_entrenamiento)

    return datos_proyectados

def getHistorialAcademico(alumno):

    class materia(BaseModel):
        id_materia: int
        nota: Optional[float]
        estado: str

    class Response(BaseModel):
        historia: list[materia]

    df = csvReader.filtrar_filas_archivo(
        ruta_archivo=ruta_archivo_regularidades,
        id_carrera=id_carrera,
        id_plan=id_plan,
        id_alumno=alumno
    )
    df.to_csv("archivo_reg.csv", index=False)

    with open("archivo_reg.csv", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        predicciones = []
        for fila in lector:

            nota_str = fila["nota"].strip()  
            nota_val = None if nota_str == "" else float(nota_str.replace(',', '.'))

            pred = materia(
                id_materia=int(fila["materia"]),
                nota=nota_val,
                estado=fila["cond_regularidad"]
            )
            predicciones.append(pred)
    return Response(historia=predicciones)


def getDatosPersonales(id_alumno):
    id_persona = None
    with open(ruta_archivo_alumnos, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        # Normalizamos los fieldnames: convertimos a minúsculas y quitamos espacios
        reader.fieldnames = [campo.strip().lower() for campo in reader.fieldnames if campo is not None]
        for row in reader:
            # Normalizamos las claves de cada fila para evitar problemas con mayúsculas/espacios
            fila = {k.strip().lower(): v.strip() for k, v in row.items() if v is not None}
            # Comparamos el id_alumno (convertido a cadena) con el valor en la fila
            if fila.get("id_alumno") == str(id_alumno):
                print(fila)
                id_persona = fila.get("id_persona")
                break

    filas_filtradas = []
    with open(ruta_archivo_datos_hist_personales, newline="", encoding="utf-8") as f_hist:
        # Leemos el CSV de datos históricos con delimitador '|'
        lector_hist = csv.DictReader(f_hist, delimiter="|")
        # Normalizamos los encabezados
        lector_hist.fieldnames = [campo.strip().lower() for campo in lector_hist.fieldnames if campo is not None]
        for fila in lector_hist:
            fila_normalizada = {k.strip().lower(): v.strip() for k, v in fila.items() if v is not None}
            if fila_normalizada.get("id_persona") == id_persona:
                filas_filtradas.append(fila_normalizada)
    filas = filas_filtradas
    if not filas:
        return None
    salida = StringIO()
    # Cambiamos el delimitador a coma para la salida
    writer = csv.DictWriter(salida, fieldnames=filas[0].keys(), delimiter=",")
    writer.writeheader()
    writer.writerows(filas)
    return salida.getvalue()

