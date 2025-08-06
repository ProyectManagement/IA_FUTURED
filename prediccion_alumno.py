from conexion import conectar_mongodb
from modelo import cargar_modelo, normalizar_documento
from bson import ObjectId
import datetime
import pandas as pd

def preparar_datos_alumno(encuesta, label_encoders):
    datos = normalizar_documento(encuesta)
    df = pd.DataFrame([datos])
    
    # Mapear binarios
    for col in ["trabaja", "padecimiento_cronico", "atencion_psicologica"]:
        if col in df.columns:
            df[col] = df[col].map({"Sí": 1, "No": 0}).fillna(0).astype(int)
    
    X_pred = df.drop(columns=["id_alumno", "abandona", "matricula"], errors="ignore")
    
    # Aplicar label encoders
    for col, le in label_encoders.items():
        if col in X_pred.columns:
            try:
                X_pred[col] = le.transform(X_pred[col].astype(str))
            except:
                X_pred[col] = 0  # Valor por defecto
    
    # Asegurar que tenemos todas las columnas que el modelo espera
    modelo, _ = cargar_modelo()
    for col in modelo.feature_names_in_:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[modelo.feature_names_in_]
    
    return X_pred

def predecir_por_matricula(matricula):
    # Cargar modelo pre-entrenado
    try:
        modelo, label_encoders = cargar_modelo()
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None
    
    # Conectar a MongoDB
    db = conectar_mongodb()
    
    # Buscar alumno
    alumno = db.alumnos.find_one({"matricula": matricula})
    if not alumno:
        print(f"No se encontró alumno con matrícula {matricula}")
        return None
    
    id_alumno = alumno["_id"]
    
    # Obtener última encuesta del alumno
    encuestas = list(db.encuestas.find({"id_alumno": id_alumno}).sort("_id", 1))
    if not encuestas:
        print(f"No se encontraron encuestas para el alumno con matrícula {matricula}")
        return None
    
    ultima_encuesta = encuestas[-1]
    
    # Preparar datos y predecir
    X_pred = preparar_datos_alumno(ultima_encuesta, label_encoders)
    riesgo_pred = modelo.predict_proba(X_pred)[0][1]  # Probabilidad de clase positiva
    riesgo = round(riesgo_pred * 100, 2)
    
    # Determinar nivel de riesgo
    if riesgo >= 80:
        motivo = "Alto riesgo: múltiples factores académicos y personales"
        recomendacion = "Asesoría académica urgente y apoyo psicológico"
    elif riesgo >= 60:
        motivo = "Riesgo medio: bajo promedio o problemas personales"
        recomendacion = "Tutoría y monitoreo continuo"
    elif riesgo >= 40:
        motivo = "Riesgo leve: dificultad para estudiar o motivación baja"
        recomendacion = "Seguimiento por tutor y actividades motivacionales"
    else:
        motivo = "Sin riesgo aparente"
        recomendacion = "Mantener seguimiento regular"
    
    # Obtener información del alumno
    try:
        grupo = db.grupos.find_one({"_id": ObjectId(alumno.get("id_grupo"))})
        nombre_grupo = grupo["nombre"] if grupo else "Desconocido"
    except Exception:
        nombre_grupo = "Desconocido"
    
    nombre_completo = " ".join(filter(None, [
        alumno.get("nombre", "").strip(),
        alumno.get("apellido_paterno", "").strip(),
        alumno.get("apellido_materno", "").strip()
    ])).strip() or "Desconocido"
    
    resultado = {
        "matricula": matricula,
        "nombre_completo": nombre_completo,
        "nombre_grupo": nombre_grupo,
        "riesgo": riesgo,
        "motivo": motivo,
        "recomendacion": recomendacion,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    # Guardar predicción en MongoDB
    filtro = {"id_alumno": id_alumno}
    update = {
        "$set": {
            **resultado,
            "id_alumno": id_alumno
        }
    }
    db.predicciones.update_one(filtro, update, upsert=True)
    
    return resultado

if __name__ == "__main__":
    matricula = input("Ingresa la matrícula del alumno para predecir riesgo: ").strip()
    resultado = predecir_por_matricula(matricula)
    if resultado:
        print("\nPredicción para el alumno:")
        for k, v in resultado.items():
            print(f"{k}: {v}")
    else:
        print("No se pudo obtener la predicción.")