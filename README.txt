/////////////Predicción de Trayectorias Académicas y Rendimiento de Grupos Bajo el Plan de Estudios 2022////////////////

Para su funcionamiento se necesita la instalacion de dependencias, estas son:
	.openai
	.pandas
	.pydantic
	.flask
	.flask_cors

para una instalacion rapida usando la consola pararse dentro de la carpeta del proyecto y ejecutar el siguiente comando:

	pip install -r requirements.txt

Es necesario python para su correcto funcionamiento, ya que se debera ejecutar un archivo .py.
Una vez instaladas las dependencias se debe ejecutar el archivo app.py con el comando
	
	python app.py

una vez se inicialize el servidor vera las siguientes salidas en la consola que indica una ejecucion correcta.

 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000

La pagina web ya es accesible desde un buscador web utilizando la direccion local http://127.0.0.1:5000 o aquella que sea indicada en la salida de la consola.
Una vez dentro podra escribir el id del alumno a proyectar. Para poder consultar que alumno puede consultar el archivo 001_alumnos.csv de la carpeta DataSource
asegurese de que sea un alumno que curse la carrera 206(ingenieria en sistemas) y en el plan de estudios 2011




