import pandas as pd
import os

# Método que filtra la tabla para reducir el tamaño de la misma
def filtrar_filas_archivo(ruta_archivo='', archivo_original=pd.DataFrame(), id_carrera='', id_plan='', materia=-1,
                          filtro_aprobado=False, filtro_obligatoria='') -> pd:
    if materia == -1:
        if os.path.exists(ruta_archivo):
            archivo_modificado = pd.read_csv(ruta_archivo, delimiter='|', low_memory=False)
            archivo_modificado = archivo_modificado[archivo_modificado['carrera'] == id_carrera]
            archivo_modificado = archivo_modificado[archivo_modificado['plan'] == id_plan]
            archivo_modificado = archivo_modificado[archivo_modificado['materia'].astype(str).str.match(
                r'^\d+$')]  # Filtrar los registros que contienen solo números en la columna 'materia'
            if filtro_obligatoria != '':
                archivo_modificado = archivo_modificado[archivo_modificado['obligatoria'] == filtro_obligatoria]
                archivo_modificado = archivo_modificado[archivo_modificado[
                                                            'nombre_materia'] != 'Inglés']  # para nuestro plan, ingles aparece como optativa y como ya la consideramos como obligatoria, no la tenemos en cuenta
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

id_carrera              = 206   # Ingenieria en Sistemas
id_plan_equivalencias   = "S95" # Plan de Estudios Nuevo
id_plan                 = "S95" # Plan de Estudios Viejo
equivalencias           = False # Variable para determinar si trabajar con equivalencias o no

# Pasamos la dirección del archivo de regularidades
ruta_archivo_regularidades = f"DataSource/002_regularidades.csv"

# Abrimos el archivo con el método creado anteriormente
archivo_regularidades = filtrar_filas_archivo(ruta_archivo=ruta_archivo_regularidades, id_carrera=id_carrera,
                                              id_plan=(id_plan_equivalencias if equivalencias else id_plan))
# Esto muestra las primeras 5 filas del DataFrame
# Si el archivo tiene delimitadores o formatos incorrectos, acá se verán
print(archivo_regularidades.head())