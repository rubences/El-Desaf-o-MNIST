import os
import json
from flask import Flask, render_template
from mnist_trainer import train_and_evaluate

app = Flask(__name__, template_folder='src')

# Filtro personalizado para serializar a JSON en la plantilla
@app.template_filter('tojson_safe')
def tojson_safe_filter(obj):
    return json.dumps(obj)

@app.route("/")
def index():
    # Obtenemos todos los datos visuales del entrenador
    model_summary, training_logs, final_accuracy, graphs_json, sample_images = train_and_evaluate()
    
    # Pasamos todo a la plantilla
    return render_template('index.html', 
                           model_summary=model_summary, 
                           training_logs=training_logs, 
                           final_accuracy=final_accuracy,
                           graphs_json=graphs_json,
                           sample_images=sample_images)

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
