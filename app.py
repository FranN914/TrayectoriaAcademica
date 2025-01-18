import main
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/project', methods=['POST'])
def project_student():
    data = request.get_json()
    id_alumno = data.get('id')

    if not isinstance(id_alumno, int):
        return jsonify({"error": "El identificador debe ser un número entero."}), 400

    result = main.evaluar_prediccion(id_alumno)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
