
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout

def train_and_evaluate():
    # Fase 1: Reconocimiento y Preparación de los Datos
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocesamiento
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Fase 2: Diseño de la Arquitectura de la Red
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(784,), activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Capturar el resumen del modelo
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    model_summary = f.getvalue()


    # Fase 3: El Proceso de Entrenamiento
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Capturar logs del entrenamiento
    class TrainingLogger(keras.callbacks.Callback):
        def __init__(self):
            self.logs = []
        def on_epoch_end(self, epoch, logs=None):
            self.logs.append(f"Epoch {epoch+1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

    training_logger = TrainingLogger()
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128,
                        validation_split=0.1,
                        callbacks=[training_logger],
                        verbose=0) # El verbose se pone a 0 para no imprimir en consola y poder capturar los logs

    # Fase 4: Evaluación y Análisis de Resultados
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    final_accuracy = f'Precisión en el conjunto de prueba: {test_acc}'

    # Visualización del Aprendizaje y guardado de los gráficos
    plt.figure(figsize=(12, 4))

    # Gráfico de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Precisión del Modelo')
    plt.legend()
    plt.savefig('static/accuracy.png')

    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida del Modelo')
    plt.legend()
    plt.savefig('static/loss.png')
    plt.close()

    return model_summary, training_logger.logs, final_accuracy
