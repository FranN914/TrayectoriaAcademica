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

    def construir_prompt(self, historia_academica, plan_2022, ejemplos_entrenamiento=None):
        """
        Construye un prompt estructurado para GPT-4 siguiendo la arquitectura del modelo.
        Si no se proporcionan ejemplos de entrenamiento, se basa únicamente en reglas y heurísticas.

        :param historia_academica: DataFrame con el historial académico del alumno.
        :param datos_personales: Diccionario con datos personales del alumno.
        :param plan_2022: DataFrame con la estructura del plan 2022.
        :param ejemplos_entrenamiento: Lista de ejemplos históricos para simular entrenamiento (opcional).
        :return: String con el prompt preparado.
        """

        ejemplos_texto = f"Antes de realizar predicciones, considera los siguientes ejemplos históricos:\n{ejemplos_entrenamiento}\n\n"

        prompt = """
        Eres un modelo de predicción académica avanzado basado en redes neuronales. Tu arquitectura tiene las siguientes características:
    
        1. **Entrada del Modelo:**
           - Datos Académicos: Historial académico del alumno bajo el plan 2011, incluyendo notas, estados (Aprobada, Recusada, Abandonada), y correlatividades.
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
              - **Predicción de Estados:** Estima si una materia será Aprobada, Recusada, o Abandonada.
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
           - "analisis_academico": Una descripción detallada de los patrones académicos observados.
           - "impacto_personal": Una evaluación del impacto de los datos personales en el desempeño académico.
           - "predicciones": Una lista de objetos con el id de cada materia, la nota final, y el estado proyectado.
           - "indice_exito": Un valor flotante que representa el éxito general del alumno.
    
        {ejemplos_texto}
    
        Ahora, realiza la predicción para este alumno:
    
        - **Historial académico del alumno (plan 2011):**
          {historia_academica}
    
        - **Estructura del plan de estudios 2022:**
          {plan_2022}
    
        Genera una predicción detallada en el formato JSON especificado.
        """.format(
            ejemplos_texto=ejemplos_texto,
            historia_academica=historia_academica.to_dict(orient="records"),
            #datos_personales=datos_personales,
            plan_2022=plan_2022.to_dict(orient="records")
        )
        return prompt

    # Función para procesar un archivo con GPT-4
    def procesar_archivo_con_gpt4(self, historia_academica, plan_2022,datos_entrenamiento):

        prompt = self.construir_prompt(historia_academica, plan_2022,datos_entrenamiento)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un asistente académico experto en simulaciones académicas."},
                {"role": "user", "content": prompt},
            ]
        )
        # Convertir la respuesta de GPT-4 en un DataFrame
        predicciones = response.choices[0].message.content
        return predicciones






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