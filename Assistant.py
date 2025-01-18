import json
import openai
import os

class Assistant:
    def __init__(self, archivo_indice_exito_academico):
        self.archivo_indice_exito_academico = archivo_indice_exito_academico

    def procesar_archivo_con_gpt4(self, historia_academica, plan_2022, datos_entrenamiento):
        """
        Función que se comunica con chatGPT para la construcción de la predicción de las notas y el índice académico.

        :param historia_academica: DataFrame con el historial académico del alumno.
        :param plan_2022: DataFrame con la estructura del plan 2022.
        :param datos_entrenamiento: Lista de ejemplos históricos para simular entrenamiento.
        :return: String con la predicción completa
        """

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
                    Eres un modelo de predicción académica avanzado basado en redes neuronales. Tu arquitectura tiene las siguientes características:
                    1. **Entrada del Modelo:**
                       - Datos Académicos: Historial académico del alumno bajo el plan 2011, incluyendo notas, estados (Aprobada, Recusada, Abandonada), y correlatividades. Tambien posee un diccionario con datos personales del alumno como , empleo, vivienda, responsabilidades familiares, acceso a tecnología, y otros factores externos.
                       - Estructura del Plan 2022: Detalles del plan, incluyendo orden de materias (año/cuatrimestre), correlatividades, y materias optativas.
                    2. **Arquitectura de la Red Neuronal:**
                       a) **Capa de Entrada:**
                          - Procesa los datos académicos y del plan 2022.
                          - Convierte variables categóricas (e.g., materias, estados) en embeddings y normaliza valores numéricos.
                       b) **Capas Ocultas Compartidas:**
                          - Una red densa con múltiples capas que aprende interacciones complejas entre las entradas.
                          - Usa activaciones ReLU y regularización L2 para evitar sobreajustes.
                       c) **Salidas Multitarea:**
                          - **Predicción de Notas:** Calcula las calificaciones finales para cada materia del plan 2022.
                          - **Predicción de Estados:** Estima si una materia será Aprobada, Recursada, o Abandonada.
                          - **Índice de Éxito:** Genera un puntaje global que representa el desempeño general del alumno.
                    3. **Función de Pérdida:**
                       - Para notas: Error cuadrático medio (MSE).
                       - Para estados: Entropía cruzada categórica.
                       - Para índice de éxito: Error absoluto medio (MAE).
                       - La pérdida total es una combinación ponderada de estas métricas.
                    4. **Restricciones:**
                       - Justifica cada decisión basada en los datos proporcionados.
                       - No simplifiques las predicciones basándote solo en una interpretación superficial.
                       - Si alguna información está incompleta, explícita los supuestos realizados para completar la predicción.
                    5. **Salida del Modelo:**
                       Devuelve un JSON estructurado con:
                       - "analisis_academico": Una descripción detallada y extensa de los patrones académicos observados.
                       - "impacto_personal": Una evaluación del impacto de los datos personales en el desempeño académico del plan de estudio 2011 y 2022
                       - "predicciones": Una lista de objetos con el id de cada materia, la nota final, y el estado proyectado de TODAS las materias del plan de estudio 2022 (Cantidad 37 materias).
                       - "indice_exito": Un valor flotante que representa el éxito general del alumno. Para su cálculo se deben seguir las siguientes directivas {archivo_indice_exito_academico} realizar un desgloce explicacion del proceso del calculo. El indice se compara entre su rendimiento de 2011 y el proyectado.
                    f"Antes de realizar predicciones, considera los siguientes ejemplos históricos:\n{ejemplos_entrenamiento}\n\n"
                        """.format(archivo_indice_exito_academico = self.archivo_indice_exito_academico,
                            ejemplos_entrenamiento = datos_entrenamiento)},

                {"role": "user", "content": """
                        Ahora, realiza la predicción para este alumno:
                        - **Historial académico del alumno (plan 2011):**
                          {historia_academica}
                        - **Estructura del plan de estudios 2022:**
                          {plan_2022}
            
                        Defina una prediccion simulando en profundidad una red neuronal. La certeza del analisis es
                        Genera una predicción detallada en el formato JSON especificado.
                        """.format(
                            historia_academica=historia_academica.to_dict(orient="records"),
                            archivo_indice_exito_academico=self.archivo_indice_exito_academico,
                            plan_2022=plan_2022.to_dict(orient="records")
                        )},
            ]
        )
        # Convertir la respuesta de GPT-4 en un DataFrame
        predicciones = response.choices[0].message.content
        return predicciones

    def procesar_archivos_y_generar_json(self, file_path, id_persona, output_json):
        """
        Realiza una prediccón utilizando chatGPT y retorna un JSON con la respuesta.

        :param file_path: dirección donde se guardará el archivo
        :param id_persona: identificador del alumno del cual se quiere realizar la predicción
        :param output_json: archivo JSON en el cual se guardaran las respuestas
        :return: JSON con las respuestas
        """
        resultados = []

        try:
            # Procesar el contenido con GPT-4
            resultado = self.procesar_archivo_con_gpt4(id_persona)
            # Agregar el resultado a la lista
            resultados.append({"archivo": os.path.basename(file_path), "resultado": resultado})

        except Exception as e:
            resultados.append({"archivo": os.path.basename(file_path), "resultado": f"Error al leer el archivo: {e}"})

        # Generar el archivo JSON
        with open(output_json, "w", encoding="utf-8") as jsonfile:
            json.dump(resultados, jsonfile, ensure_ascii=False, indent=4)

        print(f"Resultados guardados en: {output_json}")