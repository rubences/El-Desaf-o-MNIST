import os
import json
from flask import Flask, render_template

app = Flask(__name__, template_folder='src')

# Filtro personalizado para serializar a JSON en la plantilla
@app.template_filter('tojson_safe')
def tojson_safe_filter(obj):
    return json.dumps(obj)

@app.route("/")
def index():
    # --- CARGA RÁPIDA DESDE CACHÉ ---
    # Ya no se entrena. Solo se leen los resultados pre-calculados.
    with open('src/training_cache.json', 'r') as f:
        cached_results = json.load(f)
    
    # Pasamos los datos cacheados a la plantilla
    return render_template('index.html', 
                           model_summary=cached_results['model_summary'], 
                           training_logs=cached_results['training_logs'], 
                           final_accuracy=cached_results['final_accuracy'],
                           graphs_json=cached_results['graphs_json'],
                           sample_images=cached_results['sample_images'])

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
