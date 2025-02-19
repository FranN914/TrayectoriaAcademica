import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
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

if __name__ == '__main__':
    app.run(debug=True)