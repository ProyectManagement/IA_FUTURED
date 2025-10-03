from flask import Flask, jsonify
from conexion import conectar_mongodb
from modelo import predecir_riesgo
import joblib
import os

app = Flask(__name__)

# Cargar modelo entrenado
try:
    modelo = joblib.load('modelo_entrenado/modelo.pkl')
    label_encoders = joblib.load('modelo_entrenado/label_encoders.pkl')
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    modelo = None
    label_encoders = None

@app.route('/')
def home():
    return jsonify({
        "message": "Sistema de Predicción de Riesgo Académico",
        "status": "active",
        "endpoints": {
            "estado": "/estado",
            "predicciones": "/predicciones"
        }
    })

@app.route('/estado')
def estado():
    return jsonify({
        "modelo_cargado": modelo is not None,
        "aplicacion": "IA_FUTURED - Sistema de predicción"
    })

@app.route('/predicciones')
def obtener_predicciones():
    try:
        db = conectar_mongodb()
        encuestas = list(db.encuestas.find())
        
        if modelo and label_encoders:
            # Aquí deberías importar y usar tu función de predicción
            resultados = predecir_riesgo(modelo, encuestas, label_encoders)
            return jsonify({
                "total_predicciones": len(resultados),
                "predicciones": resultados[:10]  # Mostrar solo las primeras 10
            })
        else:
            return jsonify({"error": "Modelo no disponible"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
