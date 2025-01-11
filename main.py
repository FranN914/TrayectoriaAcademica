import CSVReader as csvReader
from openai import OpenAI

### Archivos a utilizar
ruta_archivo_regularidades              = f"DataSource/002_regularidades.csv"
ruta_archivo_alumnos                    = f"DataSource/001_alumnos.csv"
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
id_persona              = 54294  # Persona: Benito Juarez
equivalencias           = True   # Variable para determinar si trabajar con equivalencias o no

# Apertura de archivos
archivo_regularidades               = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_regularidades, id_carrera=id_carrera, id_plan=id_plan_nuevo)
archivo_alumnos                     = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_alumnos, id_carrera=id_carrera, id_plan=id_plan)
archivo_datos_personales            = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_personales)
archivo_datos_laborales             = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_laborales)
archivo_datos_hist_personales       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_personales)
archivo_datos_hist_laborales        = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_datos_hist_laborales)
archivo_equivalencias               = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_equivalencias)
archivo_etiquetas                   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_etiquetas)
archivo_optativas_etiquetado        = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_optativas_etiquetado)
archivo_plan_2011                   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011)
archivo_plan_2011_etiquetado        = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011_etiquetado)
archivo_plan_2011_precedencia       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2011_precedencia)
archivo_plan_2022                   = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2022)
archivo_plan_2022_precedencia       = csvReader.filtrar_filas_archivo(ruta_archivo=ruta_archivo_plan_2022_precedencia)
archivo_indice_exito_academico      = open(ruta_archivo_indice_exito_academico, 'r').read()


''' Implementación del modelo '''

# Comienza la implementación
client = OpenAI(

)
#Fin de la implementación


#Comienzo del entrenamiento
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "I will send you information about a graduate student who has completed his/her academic plan"
                       "under the 2011 scheme."
        },
        {
            "role": "system",
            "content": "I will send you information about the existing subjects in the Systems Engineering degree"
                       "program of the 2011 curriculum."
                       f"The information is obtained from a csv file: {archivo_plan_2011}",
        },
        {
            "role": "system",
            "content": "I will send you information about the existing subjects in the Systems Engineering degree"
                       "program of the 2022 curriculum."
                       f"The information is obtained from a csv file: {archivo_plan_2022}",
        },
        {
            "role": "system",
            "content": "I will send you information about the equivalencies of the subjects of the 2011 plan in the"
                       "2022 plan."
                       f"The information is obtained from a csv file: {archivo_equivalencias}",
        },
        {
            "role": "system",
            "content": "I will send you information about the labels. A label is a category that is assigned to a"
                       "subject or elective. This is in order to know in which category the student stands out the "
                       "most. When evaluating a subject taken in the 2022 plan, it is verified which"
                       "category/categories it belongs to and it can be known if it is one in which the student stands"
                       "out or not, in order to assign his/her grade."
                       f"The information is obtained from a csv file: {archivo_etiquetas}",
        },
        {
            "role": "system",
            "content": "I will send you information about the 2011 plan labeling."
                       f"The information is obtained from a csv file: {archivo_plan_2011_etiquetado}",
        },
        {
            "role": "system",
            "content": "I will send you information about the precedences required for each subject in the 2011 plan."
                       "The precedences are for the course (the subject cannot be taken without first taking other"
                       "subjects), and for the final (the subject cannot be taken without first passing the final exam"
                       "for other subjects). Dashes ('-') are used to separate the identifier for each subject, both"
                       "in precedence_course and in precedence_final."
                       f"The information is obtained from a csv file: {archivo_plan_2011_precedencia}",
        },
        {
            "role": "system",
            "content": "I will send you information about the precedences required for each subject in the 2022 plan."
                       "The precedences are for the course (the subject cannot be taken without first taking other"
                       "subjects), and for the final (the subject cannot be taken without first passing the final exam"
                       "for other subjects). Dashes ('-') are used to separate the identifier for each subject, both"
                       "in precedence_course and in precedence_final."
                       f"The information is obtained from a csv file: {archivo_plan_2022_precedencia}",
        },
        {
            "role": "system",
            "content": "I will send you information on how to calculate the academic success rate. This rate is used"
                       "to evaluate the student's performance throughout his/her career."
                       "Whenever you want to know how the index is calculated or to consult about it, refer to the"
                       "following file."
                       f"The information is obtained from a txt file: {archivo_indice_exito_academico}",
        }
    ],
    model="gpt-4o",
)
#Fin del entrenamiento


# Comienzo de la visualización de resultados
print(chat_completion.choices[0].message)