# Auto generated game parser
Como hipótesis del trabajo se plantea la generación de historias utilizando deep learning para, a partir de esto,  desarrollar un videojuego basado completamente en esta historia obtenida. Por lo tanto, el objetivo principal es buscar formas de procesar los textos para diferenciar las relaciones entre las palabras y, de este modo, poder obtener valores de interés: Entidades, Descripción, Posesiones y Acciones. Esto es sólo una de las etapas necesarias para llevar a cabo el proyecto de generar mundos visuales mediante deep learning.

En este trabajo se presenta un parser de diferentes atributos sobre un texto generado automáticamente. El objetivo es encontrar unidades mínimas que ayuden a recrear dicho texto de forma visual en formato de videojuego, analizando cada oración y su composición y buscando relaciones entre las palabras y las distintas sentencias. Se plantean cuatro valores de interés y se aplican técnicas de NLP, junto con diversos modelos pre entrenados para cumplir el objetivo.

El análisis del texto y el desarrollo de los algoritmos permitió comprender más en detalle los alcances y límites que tiene este proyecto, concluyendo en que al parsear el texto se pierde información que en muchos casos importante. A su vez, se comprendió que es necesario reevaluar las libertades brindadas en el juego y plantear nuevas limitantes que simplifiquen la cantidad de atributos a parsear del texto.

Más info podrá encontrarse en **Informe taller de python.pdf**

### Estructura del proyecto
- AutoGenGameParser.ipynb: notebook utilizado para probar la funcionalidad del parser completo y hacer mediciones de tiempo.
- Parsers: contiene los parsers utilizados.
	- ActionParser: se encarga de obtener todas las acciones que realizaron las entidades.
	- EntityParser: se encarga de obtener todas las entidades del texto recibido.
	- PossesionsParser: se encarga de obtener todas las posesiones de las entidades.
- Utils: archivo que facilita la implementación y el uso de la configuración del proyecto
- neuralcoref: libreria utilizada. Contiene redes neuronales para encontra coreferencias en el texto.

### Instalacion de requerimientos
Esto se puede hacer de dos formas:
- con requirements.txt: en este caso se ejecuta el siguiente comando para instalar todas las dependencias necesarias:
	- pip3 install -r requirements.txt
- con requirements\_only\_AGG.txt: en este caso se deberá ejecutar:
	- instalar requerimientos del proyecto: pip3 install -r requirements\_only\_AGG.txt
	- instalar requerimientos de neuralcoref: cd neuralcoref ; pip3 install -r requirements.txt

### Como ejecutar
Luego de instalar las dependencias, ingresar al notebook *AutoGenGameParser* y ejecutar todas las celdas. El proyecto tiene muchas dependencias, por lo cual requerira tiempo para que se instalen varios paquetes y modelos pre entrenados.
