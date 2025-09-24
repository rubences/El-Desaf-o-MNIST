import os
from flask import Flask, render_template
from mnist_trainer import train_and_evaluate

app = Flask(__name__, template_folder='src')

@app.route("/")
def index():
    # Entrena el modelo y obtiene los resultados
    model_summary, training_logs, final_accuracy = train_and_evaluate()
    
    # Renderiza la plantilla HTML, pasando los resultados
    return render_template('index.html', 
                           model_summary=model_summary, 
                           training_logs=training_logs, 
                           final_accuracy=final_accuracy)

def main():
    # Asegurarse de que el directorio static existe
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
