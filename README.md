# Curso de Inteligencia Artificial

Este repositorio presenta los ejercicios realizados durante el curso de inteligencia artificial, el cual tiene un enfoque en buscar entender los algoritmos  desde la forma como pensamos y eso como se modela con matematica. 
Iniciamos el curso revisando como los sistemas expertos es un motor de logica de primer orden que permite de forma sencilla agregar reglas para crear un pensamiento deductivo.  Luego entramos en el pensamiento inductivo con técnicas de machine learning como arboles de decisión (encontrar las reglas a través de los datos) y Naives-Bayes (encontrar las reglas teniendo encuenta tendencias de probabilidad). 

La segunda parte del curso inicia con el estudio del algoritmo del gradiente, para plantearnos la idea de la inteligencia como un proceso de optimización y como principal exponente la regresión lineal, Lasso o Ridge.  Con base en estos conceptos entramos a revisar las redes neuronales como una forma de modelar como funciona el hardware de nuestro cerebro. Aqui hicimos una pausa en el estudio de la AI, para especializar el curso en imagenes, comprendiendo por medio de ejercicios el procesamiento digital de imagenes y algunos de sus principales conceptos.

La ultima parte del curso nos adentramos en el tema de deep learning aplicado a imagenes, donde los estudiantes estudian diferentes modelos preentrenados de deep learning y en clase revisamos los procesos de creaciòn de modelos y transfer learning con ejercicios. Esto permite dividir el curso en dos partes , un proyecto donde aplican estos conceptos a imagenes y una revisión a las técnicas basadas en medidas de distancia (SVM,K-Nearest Neighbors, k-means, pca,..). Finalmente, sin mucho impulso llegamos a la definición de la inteligencia basado en modelos bioinspirados no tan egocentricos como algoritmos geneticos, computación evolutiva, Swarm optimization, o las colonias de hormigas.

La primera parte del curso se basa en la idea narrada en el libro [Human Compatible: Artificial Intelligence and the Problem of Control](https://www.amazon.com/-/es/Stuart-Russell/dp/0525558616) de Stuart Russell, el cual recomiendo como libro de lectura recomendado. La segunda parte es mas mi aproximación, cualquier feedback seria genial.


## **Ejercicios**

### [**Introducción a Jupyter**](https://colab.research.google.com/drive/1TXMSqCnuOOVvrRZnstAhGgrvsvzDg7gZ?usp=sharing)
Este tema proporciona una entrada a los principales conceptos de Jupyter, haciendo referencia a Python y Markdown. Incluye una seríe de ejemplos sobre el manejo de elementos básicos de python, así como el cargue de archivos en formatos JSON, CSV Y texto plano. 

### [**Sistemas Expertos**](https://colab.research.google.com/drive/11NgKXAOGzNc0VjWAvqTNQjI0x9DUhDV3?usp=sharing)
Esta sección proporciona los pasos para el desarrollo de un sistema experto, explicando cómo crear una instancia, declarar hechos y ejecutarlos. Para finalmente, desarrollar las fases de implementación y pruebas.

### [**Lógica Difusa**](https://colab.research.google.com/drive/1tyCb-yNm0_eWrq3ALHN82Hp0jy_B62cd?authuser=1)
Contiene los pasos para aplicar las reglas difusas y cómo simular las mismas.

### [**Clasificadores**](https://colab.research.google.com/drive/1LxEmJUYf7NiJ4tePIoBkxQnjpucB0cth?usp=sharing)
Se presenta un experimento de clasificación supervisada en el que se hace uso de una librería de aprendizaje autómatico, llamada scikit-learn, que principalmente usa como base 2 librerias, numpy y pandas. El tema se desarrolla a través de la creación de arrays, funciones, la forma en que se entiende la data, cómo se prepara la data y finalmente cómo se modela y se prueba.

### [**CRISP-DM Methodology**](https://colab.research.google.com/drive/1beNEtJwd-Vt9V45BA4RXWG0GI2f2ik8D?usp=sharing)
Se presenta un experimento para poder entender y preparar la data, dividir los datos de texto y entrenamiento para finalmente modelarlos, probarlos e implementarlos a través de la metodogía CRISP-DM, un método orientado a la minería de datos.

# **Técnicas de regresión**
## **Regresiones Lineales**
Contiene una serie de ejercicios de regresión lineal, un método para predecir el comportamiento de los datos, cómo comprenderlos mediante analítica descriptiva, preparar los datos, obtener el modelo y evalaurlo para finalmente implementarlo.

- ### [**Linear Regressor - regresiones Water Laboratory**](https://colab.research.google.com/drive/1L-XRUkxwSUbR2YMyWKcdmYngZUnG1IEI?usp=sharing)
- ### [**Linear Regressor - Boston price of houses**](https://colab.research.google.com/drive/1lqI-sfAxOLZJr7zD444mGKPSsjOlYJkA)
- ### [**Linear Regressor - NYC Taxi Duration**](https://colab.research.google.com/drive/1_83qraJBRh_L1IIBkLKZpHlZh6UpK2wO?usp=sharing)

# **Imágenes**
Se presentan los conceptos básicos del análisis de imágenes.

- ### [**Images**](https://colab.research.google.com/drive/1LAunWsFR-kgqw_0aqDjOcSLZuBJmP37v?usp=sharing)
  Se da una idea de cómo las imágenes se pueden representar como matrices numéricas en 2D(escala de grises) y 3D(2-D color). Se presenta una serie de ejemplos donde se aplican los atributos y métodos de matrices numpy, lectura y escritura de archivos, y conversiones de tonos y saturaciones  para facilitar procesos de segmentación.
- ### [**Filters and Convolution - Images 2**](https://colab.research.google.com/drive/1jtC1Dgb4iUPCmRTppIzHk-wRB9BvTcbw?usp=sharing)
  Se da una idea de cómo crear nuevas imágenes a partir de una imagen de entrada, siguiendo un patrón básico que consiste en recopilar los valores de los pixeles de la imagen de entrada, reducir los valores de los pixeles que le rodean y en la imagen de salida, completar el pixel correspondiente a la imagen de entrada.

- ### [**Computer Vision - Instructional Exercise**](https://colab.research.google.com/drive/1RWGmqoEQdeyh5TssoGtsXsFk8hbLGtWp#scrollTo=GSmFPgSRo72S) 
  Se presenta una ejemplo de manipulación de imágenes con OpenCV, una herramienta para editar, transformar y trabajar con imágenes.

# **Redes Neuronales**
  Se presenta una serie de ejemplos de clasificación con sci-kit, una biblioteca de aprendizaje automático y se hace uso de Tensorflow para la contrucción de una red neuronal convolucional. En el desarrollo del ejemplo se cargan una data, se prepara, se modela, se entrenan sus variables, se valida el modelo, se carga y se prueba, y finalmente se despliega en gradio, un servidor flask que tienen interfaces de entrada y salida para modelos de machine learning.
- ### [**Artificial Neural Networks**](https://colab.research.google.com/drive/1yWO2hlyrumpo711vCzvapfoGFpDB05RN?usp=sharing)
- ### [**Artificial Neural Networks - Testing Tensorflow (2.0)**](https://colab.research.google.com/drive/1h5VWt3xwrY11DPLK1Lyam29wcTnQXVXg?)

# **Transfer Learning**
Se presenta un ejercicio de Transfer Learning, una metodología de Deep Learning donde se explica cómo aumentar la data, cambiar la escala de los píxeles, realizar el cargue de un modelo base para generar un nuevo modelo y entrenarlo, para finalmente probarlo y desplegarlo. Esto se entrena previamente con un conjunto de datos de ImageNet, una base de datos de imágenes organizada. 
- ### [**Transfer Learning**](https://colab.research.google.com/drive/1qFdssydz7z2dO8NilLScCc5YU1hNj4lG?usp=sharing)

# **Clustering**

## **Hyperparameters**
Los hiperparámetros básicamente son parámetros donde su valor se utiliza para controlar el proceso de aprendizaje, pero estos no se aprenden directamente dentro de los estimadores. 

- ### [**Iris_clasiffication**](https://colab.research.google.com/drive/1TgZnHa4BIry9mfxSlMz4_1LaqkK6jLYk)
  Se presenta un ejemplo donde se entiende la data, se comprende a partir de análitica descriptiva, se prepara, se modela, se proporcionan los hiperparámetros, se prueba y se implementa.

## **KMeans**
KMeans es un algortimo no supervisado de clustering, se presenta una serie de ejemplos donde básicamente con ayuda de KMeans, se agrupan los objetos en k grupos de acuerdo a sus carácteristicas para realizar una segmentación de imágenes.
- ### [**Ejemplo de cluster**](https://colab.research.google.com/drive/1zy9Lj_NvtzCZVBkFyfiUrg-ofCJh_FBH)
- ### [**Clustering**](https://colab.research.google.com/drive/1k3wT0CD-_wP0DOwhOjBkNq4V5Oxnrkc8?usp=sharing)

## **PCA**
El PCA, análisis de componentes principales, es una técnica de aprendizaje no supervisado que se utiliza para descomponer un conjunto de datos multivariado. Básicamente, convertir un conjunto de observaciones de variables posiblemente correlacionadas en un conjunto de valores de variables sin una correlación lineal. A continuación se presenta una serie de ejemplos en los que se hace uso del algoritmo PCA.

- ### [**Reconocimiento de Vino - PCA wine**](https://colab.research.google.com/drive/1hdnx0G7BU-oO0PMSZChoj35lQJDjJWAE?usp=sharing)
- ### [**PCA Cancer**](https://colab.research.google.com/drive/1Nzus9MdceNlSyWQvdw8A-7mATSgMeMU0?usp=sharing)
- ### [**Detección de rostros - PCA faces**](https://colab.research.google.com/drive/1jZ6KNiJ6U7TXI-jEW_H93iWsubScQmot?usp=sharing)

## **Reglas de asociación**
- ### [**Association rules**](https://colab.research.google.com/drive/1FC09oPnTuZwM8dLr7W0PzcXL2sa4IGjn?usp=sharing)
  Se presenta un ejemplo donde se usa el algoritmo a priori, un algoritmo utilizado en minería de datos diseñado para operar en bases de datos transaccionales que sirven como punto de inicio para generar reglas de asociación.

## **Algoritmos genéticos**
- ### [**Genetic algorithm decision tree**](https://colab.research.google.com/drive/1PGF-LIzi0-K1sBa4XjtwFeaR3rwfj2mR?usp=sharing)
  Se presenta un ejemplo donde se realiza un ajuste de hiperparámetros con algoritmos genéticos.

-----------
