import openai
from pydantic import BaseModel

class Prediccion(BaseModel):
    """
    Clase utilizada internamente para la respuesta del modelo
    """
    id_materia: int
    nota: float
    estado: str

class Response(BaseModel):
    """
    Clase utilizada para especificar la estructura de la respuesta que dará el modelo
    """
    analisis_academico: str
    impacto_personal: str
    predicciones: list[Prediccion]
    indice_exito_2011: float
    indice_exito_2022: float
    desglose: str

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

        response = openai.beta.chat.completions.parse(
            response_format=Response,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Eres un modelo avanzado de predicción académica basado en redes neuronales. 
                    Tu tarea es analizar datos académicos del plan 2011 y proyectar resultados para el plan 2022. 
                    Considera los siguientes puntos:
    
                    1. **Entrada del Modelo:**
                       - Datos Académicos: Historial académico del alumno, incluyendo notas, estados (Aprobada, Recusada, Abandonada), y correlatividades.
                       - Datos Personales: Información externa del alumno (empleo, responsabilidades, acceso a tecnología, etc.).
                       - Plan 2022: Detalles de las materias, correlatividades, optativas y distribución por años/cuatrimestres.

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
                
                    4. **Salida del Modelo:** 
                       Devuelve un archivo JSON estructurado con:
                       - "analisis_academico": Descripción de patrones observados en el historial académico.
                       - "impacto_personal": Evaluación del impacto de los datos personales en el desempeño.
                       - "predicciones": Lista de objetos con `id_materia`, `nota` y `estado` proyectados de TODAS las materias del plan de estudio 2022 (Cantidad 37 materias).
                       - "indice_exito_2011": Puntaje global del alumno para el plan 2011. 
                       - "indice_exito_2022": Puntaje global proyectado del alumno. Justifica el cálculo detalladamente. Para su cálculo se deben seguir las siguientes directivas {archivo_indice_exito_academico} realizar un desgloce explicacion del proceso del calculo. El indice se compara entre su rendimiento de 2011 y el proyectado.
                       - "desglose": Explicación del cálculo del índice de éxito.
    
                    **Restricciones:** 
                    - Una materia se aprueba con una nota de 4 o mas
                    - El indice de exito siempre es mayor a 1.
                    - Justifica las decisiones basándote en los datos proporcionados. 
                    - Considera las etiquetas de cada materia, así puedes saber en qué categoría destaca más el alumno. Esto ayuda al momento de predecir notas al ver su categoría y tener una referencia.
                    - Si hay información incompleta, especifica los supuestos realizados.
                    - Responde solo con el archivo JSON en el formato indicado.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Realiza una predicción para este alumno:
                    - **Historial académico (plan 2011):**
                      {historia_academica.to_dict(orient="records")}
                    - **Estructura del plan 2022:**
                      {plan_2022.to_dict(orient="records")}
    
                    Usa ejemplos históricos para ajustar tus predicciones:
                    {datos_entrenamiento}
                    """
                }
            ]
        )
        # Convertir la respuesta de GPT-4 en un DataFrame
        predicciones = response.choices[0].message.parsed
        return predicciones