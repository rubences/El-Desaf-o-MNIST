import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
import base64
from contextlib import redirect_stdout
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from sklearn.metrics import confusion_matrix
from PIL import Image

def train_and_evaluate(num_epochs=5):
    # 1. Carga y preparación de datos
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.mnist.load_data()
    x_train = x_train_raw.astype('float32') / 255.0
    x_test = x_test_raw.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    y_train = keras.utils.to_categorical(y_train_raw, num_classes=10)
    y_test = keras.utils.to_categorical(y_test_raw, num_classes=10)

    # 2. Arquitectura del modelo
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(784,), activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    model_summary = f.getvalue()

    # 3. Entrenamiento
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    class TrainingLogger(keras.callbacks.Callback):
        def __init__(self):
            self.logs = []
        def on_epoch_end(self, epoch, logs=None):
            self.logs.append(f"Epoch {epoch+1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - acc: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_acc: {logs['val_accuracy']:.4f}")
    training_logger = TrainingLogger()
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=128, validation_split=0.1, callbacks=[training_logger], verbose=0)

    # 3.5. Guardar el modelo entrenado
    model.save('mnist_model.h5')

    # 4. Evaluación y predicción
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    final_accuracy = f'Precisión final en el conjunto de prueba: {test_acc*100:.2f}%'
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = y_test_raw

    # 5. Generación de gráficos Plotly
    epochs = list(range(1, num_epochs + 1))
    fig_acc = go.Figure([
        go.Scatter(x=epochs, y=history.history['accuracy'], name='Precisión (Train)', mode='lines+markers'),
        go.Scatter(x=epochs, y=history.history['val_accuracy'], name='Precisión (Val)', mode='lines+markers')
    ])
    fig_acc.update_layout(title='Evolución de la Precisión', template="plotly_white")

    fig_loss = go.Figure([
        go.Scatter(x=epochs, y=history.history['loss'], name='Pérdida (Train)', mode='lines+markers'),
        go.Scatter(x=epochs, y=history.history['val_loss'], name='Pérdida (Val)', mode='lines+markers')
    ])
    fig_loss.update_layout(title='Evolución de la Pérdida', template="plotly_white")

    # 6. Matriz de confusión
    cm = confusion_matrix(true_labels, predicted_labels)
    labels = [str(i) for i in range(10)]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        hoverongaps=False, colorscale='Blues'))
    fig_cm.update_layout(title='Matriz de Confusión', xaxis_title="Predicción", yaxis_title="Valor Real", template="plotly_white")

    graphs = {'accuracy': fig_acc, 'loss': fig_loss, 'confusion_matrix': fig_cm}
    graphs_json = json.dumps(graphs, cls=PlotlyJSONEncoder)

    # 7. Muestra de imágenes y predicciones
    sample_images = []
    num_samples = 16
    for i in range(num_samples):
        img = Image.fromarray(x_test_raw[i])
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        sample_images.append({
            "image": f"data:image/png;base64,{img_str}",
            "true_label": int(true_labels[i]),
            "predicted_label": int(predicted_labels[i])
        })

    return model_summary, training_logger.logs, final_accuracy, graphs_json, sample_images
