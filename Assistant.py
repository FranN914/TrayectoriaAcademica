import json
import openai
import os
import csv

class Assistant:
    def __init__(self, archivo_plan_2011, archivo_plan_2011_etiquetado, archivo_plan_2011_precedencia,
                 archivo_plan_2022, archivo_plan_2022_precedencia, archivo_equivalencias, archivo_etiquetas,
                 archivo_historia_academica, archivo_indice_exito_academico):
        self.archivo_plan_2011 = archivo_plan_2011
        self.archivo_plan_2011_etiquetado = archivo_plan_2011_etiquetado
        self.archivo_plan_2011_precedencia = archivo_plan_2011_precedencia
        self.archivo_plan_2022 = archivo_plan_2022
        self.archivo_plan_2022_precedencia = archivo_plan_2022_precedencia
        self.archivo_equivalencias = archivo_equivalencias
        self.archivo_etiquetas = archivo_etiquetas
        self.archivo_historia_academica = archivo_historia_academica
        self.archivo_indice_exito_academico = archivo_indice_exito_academico

    # Función para procesar un archivo con GPT-4
    def procesar_archivo_con_gpt4(self, id_persona):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Read the following file that contains information about the academic history of the"
                               "systems engineering degree, and about the subjects that each student took, along with"
                               "their grade. Do it in JSON format."
                               f"The information is obtained from a csv file: {self.archivo_historia_academica}",
                },
                {
                    "role": "user",
                    "content": f"Find the student with id {id_persona} and return the subjects he/she took, along with"
                               "his/her final grade.",
                },
                # {
                #     "role": "system",
                #     "content": "I will send you information about the existing subjects in the Systems Engineering degree"
                #                "program of the 2022 curriculum."
                #                f"The information is obtained from a csv file: {self.archivo_plan_2022}",
                # },
                # {
                #     "role": "system",
                #     "content": "I will send you information about the equivalencies of the subjects of the 2011 plan in the"
                #                "2022 plan."
                #                f"The information is obtained from a csv file: {self.archivo_equivalencias}",
                # },
                # {
                #     "role": "system",
                #     "content": "I will send you information about the labels. A label is a category that is assigned to a"
                #                "subject or elective. This is in order to know in which category the student stands out the "
                #                "most. When evaluating a subject taken in the 2022 plan, it is verified which"
                #                "category/categories it belongs to and it can be known if it is one in which the student stands"
                #                "out or not, in order to assign his/her grade."
                #                f"The information is obtained from a csv file: {self.archivo_etiquetas}",
                # },
                # {
                #     "role": "system",
                #     "content": "I will send you information about the 2011 plan labeling."
                #                f"The information is obtained from a csv file: {self.archivo_plan_2011_etiquetado}",
                # },
                # {
                #     "role": "system",
                #     "content": "I will send you information about the precedences required for each subject in the 2011 plan."
                #                "The precedences are for the course (the subject cannot be taken without first taking other"
                #                "subjects), and for the final (the subject cannot be taken without first passing the final exam"
                #                "for other subjects). Dashes ('-') are used to separate the identifier for each subject, both"
                #                "in precedence_course and in precedence_final."
                #                f"The information is obtained from a csv file: {self.archivo_plan_2011_precedencia}",
                # },
                # {
                #     "role": "system",
                #     "content": "I will send you information about the precedences required for each subject in the 2022 plan."
                #                "The precedences are for the course (the subject cannot be taken without first taking other"
                #                "subjects), and for the final (the subject cannot be taken without first passing the final exam"
                #                "for other subjects). Dashes ('-') are used to separate the identifier for each subject, both"
                #                "in precedence_course and in precedence_final."
                #                f"The information is obtained from a csv file: {self.archivo_plan_2022_precedencia}",
                # },
                # {
                #     "role": "system",
                #     "content": "I will send you information on how to calculate the academic success rate. This rate is used"
                #                "to evaluate the student's performance throughout his/her career."
                #                "Whenever you want to know how the index is calculated or to consult about it, refer to the"
                #                "following file."
                #                f"The information is obtained from a txt file: {self.archivo_indice_exito_academico}",
                # },
                # {
                #     "role": "user",
                #     "content": f"Based on the information you have for student {id_persona} in "
                #                f"file {self.archivo_historia_academica}, and all the information you have in other files, find"
                #                f"out what grades he or she got in each subject. Return "
                #                f"this to me.",
                # }
            ]
        )
        return response.choices[0].message.content

    # Función principal para procesar múltiples archivos
    def procesar_archivos_y_generar_json(self, file_path, id_persona, output_json):
        resultados = []

        try:
            # Procesar el contenido con GPT-4
            resultado = self.procesar_archivo_con_gpt4(id_persona)
            # Agregar el resultado a la lista
            resultados.append({"archivo": os.path.basename(file_path), "resultado": resultado})

        except Exception as e:
            resultados.append({"archivo": os.path.basename(file_path), "resultado": f"Error al leer el archivo: {e}"})

        # # Generar el archivo CSV
        # with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=["archivo", "resultado"])
        #     writer.writeheader()
        #     writer.writerows(resultados)

        # Generar el archivo JSON
        with open(output_json, "w", encoding="utf-8") as jsonfile:
            json.dump(resultados, jsonfile, ensure_ascii=False, indent=4)

        print(f"Resultados guardados en: {output_json}")