import json
from mnist_trainer import train_and_evaluate

print("\n--- INICIANDO PRE-CÓMPUTO DE RESULTADOS ---")
print("Se entrenará el modelo durante 20 épocas. Este proceso puede tardar varios minutos.")

# 1. Ejecutar la función de entrenamiento pesada
model_summary, training_logs, final_accuracy, graphs_json_str, sample_images = train_and_evaluate(num_epochs=20)

# 2. Empaquetar todos los resultados en un diccionario
results_to_cache = {
    "model_summary": model_summary,
    "training_logs": training_logs,
    "final_accuracy": final_accuracy,
    "graphs_json": graphs_json_str,  # Guardamos el string JSON de los gráficos directamente
    "sample_images": sample_images
}

# 3. Guardar el diccionario en un archivo de caché
cache_filepath = 'src/training_cache.json'
with open(cache_filepath, 'w') as f:
    json.dump(results_to_cache, f, indent=4)

print(f"\n¡Éxito! Los resultados han sido pre-calculados y guardados en '{cache_filepath}'.")
print("Ahora puedes iniciar el servidor web (`bash devserver.sh`) para ver los resultados al instante.")
print("--- PRE-CÓMPUTO FINALIZADO ---")
