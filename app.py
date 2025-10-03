from flask import Flask, jsonify, request
from conexion import conectar_mongodb
from modelo import predecir_riesgo
import joblib
import os
from bson import ObjectId

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
            "buscar_alumno": "/alumno/<matricula>",
            "ejemplo": "/alumno/2023001"
        }
    })

@app.route('/estado')
def estado():
    return jsonify({
        "modelo_cargado": modelo is not None,
        "aplicacion": "IA_FUTURED - Sistema de predicción"
    })

@app.route('/alumno/<matricula>')
def buscar_alumno(matricula):
    try:
        db = conectar_mongodb()
        
        # Buscar alumno por matrícula
        alumno = db.alumnos.find_one({"matricula": matricula})
        
        if not alumno:
            return jsonify({
                "error": "Alumno no encontrado",
                "matricula_buscada": matricula
            }), 404
        
        # Buscar encuesta del alumno
        encuesta = db.encuestas.find_one({"id_alumno": alumno["_id"]})
        
        if not encuesta:
            return jsonify({
                "error": "No se encontró encuesta para este alumno",
                "alumno": {
                    "nombre": f"{alumno.get('nombre', '')} {alumno.get('apellido_paterno', '')} {alumno.get('apellido_materno', '')}",
                    "matricula": alumno.get("matricula"),
                    "grupo": alumno.get("id_grupo")
                }
            }), 404
        
        # Realizar predicción para este alumno específico
        if modelo and label_encoders:
            # Aquí necesitarías adaptar tu función predecir_riesgo para un solo alumno
            resultado_prediccion = predecir_alumno_individual(modelo, encuesta, label_encoders, alumno)
            
            return jsonify({
                "alumno": {
                    "matricula": alumno.get("matricula"),
                    "nombre_completo": f"{alumno.get('nombre', '')} {alumno.get('apellido_paterno', '')} {alumno.get('apellido_materno', '')}",
                    "grupo": alumno.get("id_grupo")
                },
                "prediccion": resultado_prediccion,
                "encuesta_disponible": True
            })
        else:
            return jsonify({
                "error": "Modelo no disponible para predicciones",
                "alumno": {
                    "matricula": alumno.get("matricula"),
                    "nombre": f"{alumno.get('nombre', '')} {alumno.get('apellido_paterno', '')} {alumno.get('apellido_materno', '')}"
                }
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Función auxiliar para predecir un solo alumno
def predecir_alumno_individual(modelo, encuesta, label_encoders, alumno):
    """
    Adapta esta función según cómo funcione tu modelo
    """
    try:
        # Esto es un ejemplo - necesitas adaptarlo a tu estructura real
        from modelo import preparar_datos_individual
        
        # Si tienes una función para preparar datos individuales
        X_individual = preparar_datos_individual([encuesta], label_encoders)
        
        # Realizar predicción
        riesgo_prediccion = modelo.predict_proba(X_individual)[0]
        riesgo_porcentaje = round(riesgo_prediccion[1] * 100, 2)
        
        # Determinar nivel de riesgo
        if riesgo_porcentaje < 30:
            nivel = "Bajo"
            motivo = "El alumno muestra buen desempeño académico"
        elif riesgo_porcentaje < 70:
            nivel = "Medio" 
            motivo = "El alumno requiere atención preventiva"
        else:
            nivel = "Alto"
            motivo = "El alumno necesita intervención inmediata"
        
        return {
            "riesgo_porcentaje": riesgo_porcentaje,
            "nivel_riesgo": nivel,
            "motivo": motivo,
            "recomendacion": generar_recomendacion(nivel, alumno)
        }
        
    except Exception as e:
        return {
            "error_prediccion": str(e),
            "riesgo_porcentaje": None,
            "nivel_riesgo": "No disponible",
            "motivo": "Error en predicción"
        }

def generar_recomendacion(nivel_riesgo, alumno):
    recomendaciones = {
        "Bajo": "Continuar con el seguimiento regular",
        "Medio": "Programar tutorías académicas mensuales",
        "Alto": "Intervención inmediata del departamento académico"
    }
    return recomendaciones.get(nivel_riesgo, "Seguimiento estándar")

if __name__ == '__main__':
    app.run(debug=True)
