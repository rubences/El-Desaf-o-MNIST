import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from PIL import Image, ImageOps
import io

app = Flask(__name__, template_folder='src')

# --- CARGA DEL MODELO Y CACHÉ ---
# Cargamos el modelo una sola vez al inicio de la aplicación
model = keras.models.load_model('mnist_model.h5')

# Función para cargar los resultados de la caché de entrenamiento
def load_cached_results():
    with open('src/training_cache.json', 'r') as f:
        return json.load(f)

# Filtro personalizado para serializar a JSON en la plantilla
@app.template_filter('tojson_safe')
def tojson_safe_filter(obj):
    return json.dumps(obj)

# --- RUTAS DE LA APLICACIÓN ---

@app.route("/")
def index():
    # Carga los datos de los gráficos y los muestra como antes
    cached_results = load_cached_results()
    return render_template('index.html', 
                           model_summary=cached_results['model_summary'], 
                           training_logs=cached_results['training_logs'], 
                           final_accuracy=cached_results['final_accuracy'],
                           graphs_json=cached_results['graphs_json'],
                           sample_images=cached_results['sample_images'])

@app.route("/predict", methods=['POST'])
def predict():
    # 1. Recibir el archivo de imagen
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No se recibió ninguna imagen.'}), 400

    try:
        # 2. Procesar la imagen para que coincida con los datos de entrenamiento
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('L') # Abrir y convertir a escala de grises
        img = ImageOps.invert(img) # Invertir colores (dígito blanco sobre fondo negro)
        img = img.resize((28, 28)) # Redimensionar a 28x28 píxeles
        
        # Convertir a un array de numpy y normalizar
        img_array = np.array(img).astype('float32') / 255.0
        
        # Aplanar la imagen de (28, 28) a (784,)
        img_flattened = img_array.reshape(1, 784)

        # 3. Realizar la predicción con el modelo cargado
        prediction = model.predict(img_flattened)
        predicted_digit = int(np.argmax(prediction))

        # 4. Devolver el resultado
        return jsonify({'prediction': predicted_digit})

    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
