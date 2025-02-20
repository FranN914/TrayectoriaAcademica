import json
from flask import Flask, render_template, request, jsonify, Response as FlaskResponse
from flask_cors import CORS
from pydantic import BaseModel
from Assistant import Response
import main

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/evaluar_prediccion')
def evaluar():
    # Llamar a la función de predicción
    id_alumno = request.args.get('id_alumno')
    resultado = main.evaluar_prediccion(int(id_alumno))  # Se obtiene el resultado
   
    # Si el objeto tiene el método dict(), lo convertimos a un diccionario.
    if hasattr(resultado, "dict"):
        resultado = resultado.dict()
    # Guarda el JSON en un archivo para inspección
    with open("resultado_debug.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    # Devuelve la respuesta en JSON
    return jsonify(resultado)

@app.route('/get_historial_academico')
def getHistorialAcademico():
    id_alumno = request.args.get('id_alumno')
    resultado = main.getHistorialAcademico(int(id_alumno))
    if hasattr(resultado, "dict"):
        resultado = resultado.dict()
    return jsonify(resultado)


@app.route('/get_datos_personales')
def get_datos_personales_endpoint():
    id_alumno = request.args.get("id_alumno")
    if not id_alumno:
        return jsonify({"error": "No se proporcionó id_alumno"}), 400

    try:
        resultado = main.getDatosPersonales(int(id_alumno))
    except Exception as e:
        return jsonify({"error": f"Error al obtener datos personales: {e}"}), 500

    # Retornamos el CSV usando la clase Response de Flask, alias FlaskResponse
    return FlaskResponse(resultado, mimetype="text/csv")
if __name__ == '__main__':
    app.run(debug=True)