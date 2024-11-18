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