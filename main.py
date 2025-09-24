import os
from flask import Flask, send_file, render_template_string
from mnist_trainer import train_and_evaluate

app = Flask(__name__)

@app.route("/")
def index():
    return send_file('src/index.html')

@app.route("/train")
def train():
    model_summary, training_logs, final_accuracy = train_and_evaluate()
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resultados del Entrenamiento</title>
    </head>
    <body>
        <h1>Resultados del Entrenamiento de MNIST</h1>
        
        <h2>Resumen del Modelo</h2>
        <pre>{{ model_summary }}</pre>

        <h2>Logs de Entrenamiento</h2>
        <pre>
        {% for log in training_logs %}
            {{ log }}
        {% endfor %}
        </pre>

        <h2>Evaluación Final</h2>
        <p>{{ final_accuracy }}</p>

        <h2>Gráficos de Entrenamiento</h2>
        <img src="/static/accuracy.png" alt="Gráfico de Precisión">
        <img src="/static/loss.png" alt="Gráfico de Pérdida">
    </body>
    </html>
    """, model_summary=model_summary, training_logs=training_logs, final_accuracy=final_accuracy)

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
