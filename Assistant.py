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
                    "content": "Always return a JSON with what is required. Never add extra text explaining. You must"
                               "follow the same structure as the 'DataSource/ejemplo.json' file. First, always the"
                               "categories (The key must be the name of the category). Inside, the subjects and the"
                               "average for the category. Inside, its identifier, name and grade.",
                },
                {
                    "role": "system",
                    "content": "Read the following file that contains information about the academic history of the"
                               "systems engineering degree, and about the subjects that each student took, along with"
                               "their grade. Do it in JSON format."
                               f"The information is obtained from a csv file: {self.archivo_historia_academica}",
                },
                {
                    "role": "system",
                    "content": "Using the subject ID obtained from the provided student, finds which category each"
                               "subject belongs to and returns a JSON with the category name and its average."
                               f"The information is obtained from a csv file: {self.archivo_plan_2011_etiquetado}",
                },
                {
                    "role": "system",
                    "content": f"Using the identifier of each subject in the 2011 plan, look for the equivalency"
                               f"identifier in the {self.archivo_equivalencias} file, and then use this identifier to"
                               f"obtain the equivalent subject in the 2022 plan in the {self.archivo_plan_2022} file."
                               f"There may be more than one subject from the 2011 plan with the same identifier for"
                               f"the 2022 plan. They will be separated by '-'."
                               f"If no identifier is found, represent the subject as an empty string, and its grade"
                               f"will be 0."
                               f"The information is obtained from a csv file: {self.archivo_equivalencias}. ",
                },
                {
                    "role": "user",
                    "content": f"Search for the student with id {id_persona} and also return the equivalent subject in the 2022 plan.",
                },
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