# Desafío MNIST: Entrenamiento de una Red Neuronal

Este proyecto documenta el proceso de construcción, entrenamiento y evaluación de una red neuronal profunda para el reconocimiento de dígitos escritos a mano del dataset MNIST, utilizando Python, TensorFlow y Keras.

## Conclusiones del Entrenamiento

La precisión final del modelo en el conjunto de datos de prueba fue de aproximadamente **97.76%**. A continuación se presenta un análisis detallado de los resultados.

### Análisis de Variaciones en los Datos

Al examinar el dataset MNIST, se identificaron variaciones significativas en la caligrafía de un mismo dígito. Por ejemplo, la inclinación, el grosor del trazo y la forma (como la presencia o ausencia de un bucle en el número "2") cambian entre imágenes. Estas variaciones constituyen un desafío para cualquier algoritmo, ya que debe aprender a **generalizar** las características esenciales de cada dígito en lugar de memorizar ejemplos específicos.

### Detección de Sobreajuste (Overfitting)

Los gráficos de entrenamiento revelaron claros signos de sobreajuste:

- **Precisión del Modelo**: Mientras que la precisión en el conjunto de entrenamiento alcanzó casi el 100%, la precisión en el conjunto de validación se estancó alrededor del 98% y dejó de mejorar en las últimas épocas de entrenamiento.
- **Pérdida del Modelo**: La pérdida de entrenamiento disminuyó de manera constante, pero la pérdida de validación comenzó a aumentar a partir de la sexta época. Esta divergencia es un indicador clásico de que el modelo está comenzando a memorizar el ruido de los datos de entrenamiento en lugar de generalizar.

### ¿Qué es el Sobreajuste?

El sobreajuste ocurre cuando un modelo se especializa tanto en los datos de entrenamiento que pierde su capacidad de hacer predicciones precisas sobre datos nuevos y no vistos. Es problemático porque el objetivo de un modelo de machine learning es su rendimiento en el mundo real, no su desempeño con datos que ya conoce. Un modelo sobreajustado no es fiable para tareas de producción.

## Fases del Entrenamiento

A continuación, se muestran los resultados detallados de cada fase del entrenamiento, incluyendo las métricas y los gráficos generados.
