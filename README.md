# Investigacion y Desarrollo de Algoritmos de Clasificacion en Machine Learning Supervisado

## Parte 1: Algoritmos de Clasificacion en ML Supervisado

### 1. Que es la clasificacion en Machine Learning y cual es su proposito?

La clasificacion en Machine Learning supervisado es una tecnica que permite entrenar un modelo usando datos previamente etiquetados para que aprenda a asignar nuevos ejemplos a una categoria o clase.

Su proposito es predecir a que grupo pertenece un dato nuevo. A diferencia de la regresion, que predice valores numericos continuos, la clasificacion predice categorias.

Un ejemplo practico seria detectar si un correo electronico es spam o no spam. El modelo se entrena con correos ya clasificados y aprende patrones como palabras sospechosas, remitentes o enlaces. Despues, cuando llega un correo nuevo, puede clasificarlo automaticamente.

### 2. Diferencia entre clasificacion binaria y multiclase

La clasificacion binaria ocurre cuando solo existen dos clases posibles. Por ejemplo, predecir si un estudiante aprueba o no aprueba un examen. Se usa cuando el problema tiene una respuesta de tipo si/no, verdadero/falso o positivo/negativo.

La clasificacion multiclase ocurre cuando existen mas de dos clases posibles. Por ejemplo, clasificar una noticia como deportes, politica, tecnologia, economia o cultura. Se usa cuando cada observacion debe pertenecer a una categoria entre varias opciones.

### 3. Principales algoritmos de clasificacion supervisada

#### Regresion Logistica

La regresion logistica es un algoritmo de clasificacion que predice la probabilidad de que un dato pertenezca a una clase. Aunque su nombre contiene la palabra "regresion", se usa principalmente para clasificacion binaria. Utiliza una funcion sigmoide para convertir el resultado en una probabilidad entre 0 y 1.

#### Arboles de Decision

Los arboles de decision clasifican datos mediante una estructura similar a un arbol. Cada nodo representa una pregunta sobre una caracteristica, cada rama representa una posible respuesta y cada hoja representa una clase final. Son faciles de interpretar, aunque pueden sobreajustarse si no se controlan.

#### Support Vector Machine (SVM)

SVM busca encontrar la mejor frontera de decision que separa las clases. En problemas simples, esta frontera puede ser una linea; en problemas mas complejos, puede usar funciones kernel para separar datos que no son linealmente separables. Es util cuando se busca una separacion clara entre clases.

#### K-Nearest Neighbors (KNN)

KNN clasifica un nuevo dato observando los ejemplos mas cercanos dentro del conjunto de entrenamiento. La clase se decide por votacion entre sus vecinos mas proximos. Es un algoritmo sencillo, pero puede volverse lento con muchos datos.

#### Naive Bayes

Naive Bayes es un algoritmo basado en el teorema de Bayes. Calcula la probabilidad de que un dato pertenezca a una clase segun sus caracteristicas. Se llama "naive" porque asume que las caracteristicas son independientes entre si. Es muy usado en clasificacion de textos, como deteccion de spam.

### 4. Metricas para evaluar modelos de clasificacion

#### Accuracy (Exactitud)

La exactitud mide la proporcion de predicciones correctas sobre el total de predicciones realizadas.

Formula:

```text
Accuracy = predicciones correctas / total de predicciones
```

Es una metrica facil de entender, pero puede ser enganosa cuando las clases estan desbalanceadas. Por ejemplo, si el 95% de los correos no son spam, un modelo que siempre predice "no spam" tendria alta exactitud, aunque no detecte correctamente el spam.

#### Precision (Precision)

La precision mide cuantas de las predicciones positivas hechas por el modelo fueron realmente positivas.

Formula:

```text
Precision = verdaderos positivos / (verdaderos positivos + falsos positivos)
```

Es importante cuando queremos evitar falsos positivos. Por ejemplo, en un sistema medico, una precision alta ayuda a evitar diagnosticar una enfermedad cuando la persona no la tiene.

#### Recall (Sensibilidad)

El recall mide cuantos casos positivos reales fueron detectados correctamente por el modelo.

Formula:

```text
Recall = verdaderos positivos / (verdaderos positivos + falsos negativos)
```

Es importante cuando queremos evitar falsos negativos. Por ejemplo, en deteccion de enfermedades, interesa detectar la mayor cantidad posible de pacientes enfermos.

#### F1-Score

El F1-Score combina precision y recall en una sola metrica. Es la media armonica entre ambas.

Formula:

```text
F1-Score = 2 * (precision * recall) / (precision + recall)
```

Se usa cuando se necesita equilibrio entre precision y recall, especialmente en conjuntos de datos desbalanceados.

#### Matriz de Confusion

La matriz de confusion es una tabla que muestra los aciertos y errores del modelo. En clasificacion binaria contiene:

- Verdaderos positivos: casos positivos correctamente clasificados.
- Verdaderos negativos: casos negativos correctamente clasificados.
- Falsos positivos: casos negativos clasificados como positivos.
- Falsos negativos: casos positivos clasificados como negativos.

Esta matriz permite entender mejor en que tipo de errores esta fallando el modelo.

#### ROC-AUC

La curva ROC muestra la relacion entre la tasa de verdaderos positivos y la tasa de falsos positivos para diferentes umbrales de decision.

El valor AUC representa el area bajo esa curva. Un AUC cercano a 1 indica un buen modelo, mientras que un AUC cercano a 0.5 indica que el modelo se comporta casi como una prediccion aleatoria.

### 5. Feature engineering y feature selection en clasificacion

El feature engineering consiste en crear, transformar o mejorar variables para que el modelo pueda aprender mejor. Por ejemplo, si tenemos la fecha de nacimiento de una persona, podemos transformarla en edad, que suele ser una variable mas util para el modelo.

El feature selection consiste en seleccionar las variables mas relevantes para entrenar el modelo y eliminar aquellas que aportan poco, generan ruido o aumentan innecesariamente la complejidad.

Ambos procesos son importantes porque la calidad de las caracteristicas influye directamente en el rendimiento del modelo.

### 7. Hiperparametros y optimizacion

Los hiperparametros son configuraciones del modelo que se definen antes del entrenamiento. No se aprenden automaticamente a partir de los datos, sino que los establece la persona que entrena el modelo.

Ejemplos de hiperparametros:

- En KNN, el numero de vecinos `k`.
- En arboles de decision, la profundidad maxima del arbol.
- En SVM, el parametro `C` o el tipo de kernel.
- En regresion logistica, la regularizacion.

Para ajustar hiperparametros se pueden usar tecnicas como Grid Search y Random Search.

#### Grid Search

Grid Search prueba todas las combinaciones posibles de hiperparametros definidas previamente. Por ejemplo, si queremos probar varios valores de `k` en KNN y varias metricas de distancia, Grid Search entrena y evalua el modelo con cada combinacion.

Su ventaja es que revisa todas las opciones indicadas, pero puede ser lento si hay muchas combinaciones.

#### Random Search

Random Search prueba combinaciones aleatorias de hiperparametros dentro de un rango definido. No revisa todas las combinaciones, pero suele ser mas rapido que Grid Search y puede encontrar buenos resultados con menos pruebas.

Se usa especialmente cuando hay muchos hiperparametros o cuando el espacio de busqueda es muy grande.

## Parte 2: Practica

La practica se desarrollara en un notebook de Jupyter usando regresion logistica para predecir si un estudiante aprobara un examen segun las horas de estudio.
