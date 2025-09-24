import os
from flask import Flask, render_template
from mnist_trainer import train_and_evaluate

# Indicamos que 'src' es la carpeta de plantillas
app = Flask(__name__, template_folder='src')

@app.route("/")
def index():
    # Entrenamos el modelo y obtenemos todos los resultados, incluyendo el JSON de los gráficos
    model_summary, training_logs, final_accuracy, graphs_json = train_and_evaluate()
    
    # Renderizamos la plantilla, pasando todos los datos necesarios
    return render_template('index.html', 
                           model_summary=model_summary, 
                           training_logs=training_logs, 
                           final_accuracy=final_accuracy,
                           graphs_json=graphs_json)

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
