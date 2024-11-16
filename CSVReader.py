import pandas as pd
import os

# Método que filtra la tabla para reducir el tamaño de la misma
def filtrar_filas_archivo(ruta_archivo='', archivo_original=pd.DataFrame(), id_carrera='', id_plan='', materia=-1,
                          filtro_aprobado=False, filtro_obligatoria='', id_persona='') -> pd:
    if materia == -1:
        if os.path.exists(ruta_archivo):
            archivo_modificado = pd.read_csv(ruta_archivo, delimiter='|', low_memory=False)
            if id_carrera != '':    # Si se pasó el parámetro id_carrera
                archivo_modificado = archivo_modificado[archivo_modificado['carrera'] == id_carrera]
            if id_plan != '':       # Si se pasó el parámetro id_plan
                archivo_modificado = archivo_modificado[archivo_modificado['plan'] == id_plan]
            if 'materia' in archivo_modificado.columns:
                # Filtrar los registros que contienen solo números en la columna 'materia'
                archivo_modificado = archivo_modificado[archivo_modificado['materia'].astype(str).str.match(r'^\d+$')]
            if id_persona != '':    # Si se pasó el parámetro id_persona
                archivo_modificado = archivo_modificado[archivo_modificado['id_persona'] == id_persona]
            if filtro_obligatoria != '':
                archivo_modificado = archivo_modificado[archivo_modificado['obligatoria'] == filtro_obligatoria]
                # Para nuestro plan, inglés aparece como optativa y como ya la consideramos como obligatoria, no la tenemos en cuenta
                archivo_modificado = archivo_modificado[archivo_modificado['nombre_materia'] != 'Inglés']
        else:
            return pd.DataFrame()
    else:
        archivo_modificado = archivo_original[archivo_original['materia'].astype(int) == int(materia)]
        if filtro_aprobado:
            archivo_modificado = archivo_modificado[archivo_modificado['resultado'] == 'A']
    return archivo_modificado

# Método que obtiene por cada alumno la fecha menor de inscripcion a una determinada cursada
def obtener_fechas_inscripcion(archivo_regularidades, materia):
    alumnos = []
    archivo_regularidades = filtrar_filas_archivo(archivo_original=archivo_regularidades, materia=materia)

    for indice, fila in archivo_regularidades.iterrows():
        existe_alumno = False
        for a in alumnos:
            if fila['id_alumno'] == a['id_alumno']:
                existe_alumno = True
                if fila['fecha_regularidad'] < a['fecha_regularidad']:
                    a['fecha_regularidad'] = fila['fecha_regularidad']

        if not existe_alumno:
            alumno = {}
            alumno['id_alumno'] = fila['id_alumno']
            alumno['fecha_regularidad'] = fila['fecha_regularidad']
            alumnos.append(alumno)
    return alumnos

# Pasamos la dirección de los archivos
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
id_carrera              = 206   # Ingenieria en Sistemas
id_plan_equivalencias   = "S95" # Plan de Estudios Nuevo
id_plan                 = "S95" # Plan de Estudios Viejo
id_persona              = 54294 # Persona: Benito Juarez
equivalencias           = False # Variable para determinar si trabajar con equivalencias o no

# Apertura de archivos
archivo_regularidades = filtrar_filas_archivo(ruta_archivo=ruta_archivo_regularidades, id_carrera=id_carrera,
                                              id_plan=(id_plan_equivalencias if equivalencias else id_plan))
archivo_materias = filtrar_filas_archivo(ruta_archivo=ruta_archivo_materias, id_carrera=id_carrera,
                                              id_plan=(id_plan_equivalencias if equivalencias else id_plan))
archivo_alumnos = filtrar_filas_archivo(ruta_archivo=ruta_archivo_alumnos, id_carrera=id_carrera,
                                              id_plan=(id_plan_equivalencias if equivalencias else id_plan))
archivo_datos_personales = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_personales, id_persona=id_persona)
archivo_datos_laborales = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_laborales, id_persona=id_persona)
archivo_datos_actividades = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_actividades, id_persona=id_persona)
archivo_datos_docente = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_docente, id_persona=id_persona)
archivo_datos_discapacidad = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_discapacidad, id_persona=id_persona)
archivo_datos_hist_personales = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_personales, id_persona=id_persona)
archivo_datos_hist_laborales = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_laborales, id_persona=id_persona)
archivo_datos_hist_actividades = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_actividades, id_persona=id_persona)
archivo_datos_hist_discapacidad = filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_discapacidad, id_persona=id_persona)


# Esto muestra las primeras 5 filas del DataFrame
# Si el archivo tiene delimitadores o formatos incorrectos, acá se verán

# print(archivo_regularidades.head())
# print(archivo_materias.head())
# print(archivo_alumnos.head())
# print(archivo_datos_personales.head())
# print(archivo_datos_laborales.head())
# print(archivo_datos_actividades.head())
# print(archivo_datos_docente.head())
# print(archivo_datos_discapacidad.head())
# print(archivo_datos_hist_personales.head())
# print(archivo_datos_hist_laborales.head())
# print(archivo_datos_hist_actividades.head())
print(archivo_datos_hist_discapacidad.head())