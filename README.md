# Investigación y Desarrollo de Algoritmos de Clasificación en Machine Learning Supervisado

## Parte 1: Algoritmos de Clasificación en ML Supervisado

### 1. ¿Qué es la clasificación en Machine Learning y cuál es su propósito?

La clasificación en Machine Learning supervisado es una técnica que permite entrenar un modelo usando datos previamente etiquetados para que aprenda a asignar nuevos ejemplos a una categoría o clase.

Su propósito es predecir a qué grupo pertenece un dato nuevo. A diferencia de la regresión, que predice valores numéricos continuos, la clasificación predice categorías.

Un ejemplo práctico sería detectar si un correo electrónico es spam o no spam. El modelo se entrena con correos ya clasificados y aprende patrones como palabras sospechosas, remitentes o enlaces. Después, cuando llega un correo nuevo, puede clasificarlo automáticamente.

### 2. Diferencia entre clasificación binaria y multiclase

La clasificación binaria ocurre cuando solo existen dos clases posibles. Por ejemplo, predecir si un estudiante aprueba o no aprueba un examen. Se usa cuando el problema tiene una respuesta de tipo sí/no, verdadero/falso o positivo/negativo.

La clasificación multiclase ocurre cuando existen más de dos clases posibles. Por ejemplo, clasificar una noticia como deportes, política, tecnología, economía o cultura. Se usa cuando cada observación debe pertenecer a una categoría entre varias opciones.

### 3. Principales algoritmos de clasificación supervisada

#### Regresión Logística

La regresión logística es un algoritmo de clasificación que predice la probabilidad de que un dato pertenezca a una clase. Aunque su nombre contiene la palabra "regresión", se usa principalmente para clasificación binaria. Utiliza una función sigmoide para convertir el resultado en una probabilidad entre 0 y 1.

#### Árboles de Decisión

Los árboles de decisión clasifican datos mediante una estructura similar a un árbol. Cada nodo representa una pregunta sobre una característica, cada rama representa una posible respuesta y cada hoja representa una clase final. Son fáciles de interpretar, aunque pueden sobreajustarse si no se controlan.

#### Support Vector Machine (SVM)

SVM busca encontrar la mejor frontera de decisión que separa las clases. En problemas simples, esta frontera puede ser una línea; en problemas más complejos, puede usar funciones kernel para separar datos que no son linealmente separables. Es útil cuando se busca una separación clara entre clases.

#### K-Nearest Neighbors (KNN)

KNN clasifica un nuevo dato observando los ejemplos más cercanos dentro del conjunto de entrenamiento. La clase se decide por votación entre sus vecinos más próximos. Es un algoritmo sencillo, pero puede volverse lento con muchos datos.

#### Naive Bayes

Naive Bayes es un algoritmo basado en el teorema de Bayes. Calcula la probabilidad de que un dato pertenezca a una clase según sus características. Se llama "naive" porque asume que las características son independientes entre sí. Es muy usado en clasificación de textos, como detección de spam.

### 4. Métricas para evaluar modelos de clasificación

#### Accuracy (Exactitud)

La exactitud mide la proporción de predicciones correctas sobre el total de predicciones realizadas.

Fórmula:

```text
Accuracy = predicciones correctas / total de predicciones
```

Es una métrica fácil de entender, pero puede ser engañosa cuando las clases están desbalanceadas. Por ejemplo, si el 95% de los correos no son spam, un modelo que siempre predice "no spam" tendría alta exactitud, aunque no detecte correctamente el spam.

#### Precision (Precisión)

La precisión mide cuántas de las predicciones positivas hechas por el modelo fueron realmente positivas.

Fórmula:

```text
Precision = verdaderos positivos / (verdaderos positivos + falsos positivos)
```

Es importante cuando queremos evitar falsos positivos. Por ejemplo, en un sistema médico, una precisión alta ayuda a evitar diagnosticar una enfermedad cuando la persona no la tiene.

#### Recall (Sensibilidad)

El recall mide cuántos casos positivos reales fueron detectados correctamente por el modelo.

Fórmula:

```text
Recall = verdaderos positivos / (verdaderos positivos + falsos negativos)
```

Es importante cuando queremos evitar falsos negativos. Por ejemplo, en detección de enfermedades, interesa detectar la mayor cantidad posible de pacientes enfermos.

#### F1-Score

El F1-Score combina precisión y recall en una sola métrica. Es la media armónica entre ambas.

Fórmula:

```text
F1-Score = 2 * (precision * recall) / (precision + recall)
```

Se usa cuando se necesita equilibrio entre precisión y recall, especialmente en conjuntos de datos desbalanceados.

#### Matriz de Confusión

La matriz de confusión es una tabla que muestra los aciertos y errores del modelo. En clasificación binaria contiene:

- Verdaderos positivos: casos positivos correctamente clasificados.
- Verdaderos negativos: casos negativos correctamente clasificados.
- Falsos positivos: casos negativos clasificados como positivos.
- Falsos negativos: casos positivos clasificados como negativos.

Esta matriz permite entender mejor en qué tipo de errores está fallando el modelo.

#### ROC-AUC

La curva ROC muestra la relación entre la tasa de verdaderos positivos y la tasa de falsos positivos para diferentes umbrales de decisión.

El valor AUC representa el área bajo esa curva. Un AUC cercano a 1 indica un buen modelo, mientras que un AUC cercano a 0.5 indica que el modelo se comporta casi como una predicción aleatoria.

### 5. Feature engineering y feature selection en clasificación

El feature engineering consiste en crear, transformar o mejorar variables para que el modelo pueda aprender mejor. Por ejemplo, si tenemos la fecha de nacimiento de una persona, podemos transformarla en edad, que suele ser una variable más útil para el modelo.

El feature selection consiste en seleccionar las variables más relevantes para entrenar el modelo y eliminar aquellas que aportan poco, generan ruido o aumentan innecesariamente la complejidad.

Ambos procesos son importantes porque la calidad de las características influye directamente en el rendimiento del modelo.

### 7. Hiperparámetros y optimización

Los hiperparámetros son configuraciones del modelo que se definen antes del entrenamiento. No se aprenden automáticamente a partir de los datos, sino que los establece la persona que entrena el modelo.

Ejemplos de hiperparámetros:

- En KNN, el número de vecinos `k`.
- En árboles de decisión, la profundidad máxima del árbol.
- En SVM, el parámetro `C` o el tipo de kernel.
- En regresión logística, la regularización.

Para ajustar hiperparámetros se pueden usar técnicas como Grid Search y Random Search.

#### Grid Search

Grid Search prueba todas las combinaciones posibles de hiperparámetros definidas previamente. Por ejemplo, si queremos probar varios valores de `k` en KNN y varias métricas de distancia, Grid Search entrena y evalúa el modelo con cada combinación.

Su ventaja es que revisa todas las opciones indicadas, pero puede ser lento si hay muchas combinaciones.

#### Random Search

Random Search prueba combinaciones aleatorias de hiperparámetros dentro de un rango definido. No revisa todas las combinaciones, pero suele ser más rápido que Grid Search y puede encontrar buenos resultados con menos pruebas.

Se usa especialmente cuando hay muchos hiperparámetros o cuando el espacio de búsqueda es muy grande.

## Parte 2: Práctica

La práctica se desarrollará en un notebook de Jupyter usando regresión logística para predecir si un estudiante aprobará un examen según las horas de estudio.
