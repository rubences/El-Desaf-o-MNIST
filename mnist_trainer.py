
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
from contextlib import redirect_stdout
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

def train_and_evaluate():
    # Fase 1: Datos
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Fase 2: Arquitectura
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(784,), activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Capturar resumen del modelo
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    model_summary = f.getvalue()

    # Fase 3: Entrenamiento
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    class TrainingLogger(keras.callbacks.Callback):
        def __init__(self):
            self.logs = []
        def on_epoch_end(self, epoch, logs=None):
            self.logs.append(f"Epoch {epoch+1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - acc: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_acc: {logs['val_accuracy']:.4f}")

    training_logger = TrainingLogger()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1, callbacks=[training_logger], verbose=0)

    # Fase 4: Evaluación
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    final_accuracy = f'Precisión final en el conjunto de prueba: {test_acc*100:.2f}%'

    # Generar JSON para gráficos Plotly
    epochs = list(range(1, 21))
    
    # Gráfico de Precisión
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=history.history['accuracy'], mode='lines+markers', name='Precisión de Entrenamiento'))
    fig_acc.add_trace(go.Scatter(x=epochs, y=history.history['val_accuracy'], mode='lines+markers', name='Precisión de Validación'))
    fig_acc.update_layout(title='Evolución de la Precisión', xaxis_title='Época', yaxis_title='Precisión', legend=dict(x=0.01, y=0.99), template="plotly_white")

    # Gráfico de Pérdida
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=history.history['loss'], mode='lines+markers', name='Pérdida de Entrenamiento'))
    fig_loss.add_trace(go.Scatter(x=epochs, y=history.history['val_loss'], mode='lines+markers', name='Pérdida de Validación'))
    fig_loss.update_layout(title='Evolución de la Pérdida', xaxis_title='Época', yaxis_title='Pérdida', legend=dict(x=0.01, y=0.99), template="plotly_white")
    
    # Convertir a JSON
    graphs_json = json.dumps({'accuracy': fig_acc, 'loss': fig_loss}, cls=PlotlyJSONEncoder)

    return model_summary, training_logger.logs, final_accuracy, graphs_json
