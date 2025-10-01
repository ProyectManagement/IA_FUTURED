import sys
import os
import json
import joblib
import pandas as pd
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from conexion import conectar_mongodb
from bson import ObjectId
from datetime import datetime, timezone

MODEL_FILE = "modelo_global.pkl"
ENCODERS_FILE = "label_encoders.pkl"

def normalizar_documento(doc):
    return {
        "id_alumno": doc.get("id_alumno"),
        "matricula": doc.get("matricula"),
        "trabaja": doc.get("aspectos_socioeconomicos", {}).get("trabaja"),
        "ingreso_mensual": doc.get("aspectos_socioeconomicos", {}).get("ingreso_mensual"),
        "padecimiento_cronico": doc.get("condiciones_salud", {}).get("padecimiento_cronico"),
        "atencion_psicologica": doc.get("condiciones_salud", {}).get("atencion_psicologica"),
        "horas_sueno": doc.get("condiciones_salud", {}).get("horas_sueno"),
        "alimentacion": doc.get("condiciones_salud", {}).get("alimentacion"),
        "materias_reprobadas": doc.get("analisis_academico", {}).get("materias_reprobadas"),
        "promedio_previo": doc.get("analisis_academico", {}).get("promedio_previo"),
        "motivacion": doc.get("analisis_academico", {}).get("motivacion"),
        "dificultad_estudio": doc.get("analisis_academico", {}).get("dificultad_estudio"),
        "expectativa_terminar": doc.get("analisis_academico", {}).get("expectativa_terminar"),
        "abandona": doc.get("abandona", "No")
    }

def preparar_datos_regresion(encuestas):
    datos = [normalizar_documento(doc) for doc in encuestas]
    df = pd.DataFrame(datos)

    required = [
        "trabaja", "ingreso_mensual", "padecimiento_cronico",
        "atencion_psicologica", "materias_reprobadas", "promedio_previo",
        "motivacion", "dificultad_estudio", "expectativa_terminar", "abandona"
    ]
    df = df.dropna(subset=required)
    if df.empty:
        raise RuntimeError("No hay encuestas válidas después de filtrar datos faltantes.")

    for col in ["trabaja", "padecimiento_cronico", "atencion_psicologica"]:
        df[col] = df[col].map({"Sí": 1, "No": 0}).fillna(0).astype(int)

    y = df["abandona"].map(lambda x: 1 if x == "Sí" else 0)
    X = df.drop(columns=["id_alumno", "abandona", "matricula"], errors="ignore")

    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, y, label_encoders

def preparar_una_encuesta(encuesta, label_encoders):
    datos = normalizar_documento(encuesta)
    df = pd.DataFrame([datos])

    for col in ["trabaja", "padecimiento_cronico", "atencion_psicologica"]:
        if col in df.columns:
            df[col] = df[col].map({"Sí": 1, "No": 0}).fillna(0).astype(int)

    X_pred = df.drop(columns=["id_alumno", "abandona", "matricula"], errors="ignore")

    for col, le in label_encoders.items():
        if col in X_pred.columns:
            raw = X_pred.at[0, col]
            val = str(raw) if pd.notna(raw) else "NA"
            if val not in le.classes_:
                if "NA" in le.classes_:
                    val = "NA"
                else:
                    val = le.classes_[0]
            try:
                X_pred[col] = le.transform([val])[0]
            except Exception:
                X_pred[col] = 0

    for col in X_pred.columns:
        if pd.api.types.is_object_dtype(X_pred[col]):
            try:
                X_pred[col] = pd.to_numeric(X_pred[col], errors="coerce").fillna(0)
            except:
                X_pred[col] = 0

    return X_pred

def cargar_o_entrenar(db):
    modelo = None
    label_encoders = None

    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODERS_FILE):
        try:
            modelo = joblib.load(MODEL_FILE)
            label_encoders = joblib.load(ENCODERS_FILE)
        except Exception:
            modelo = None
            label_encoders = None

    if modelo is not None and label_encoders is not None:
        return modelo, label_encoders

    encuestas = list(db.encuestas.find())
    if not encuestas:
        raise RuntimeError("No hay encuestas en la base de datos para entrenar.")

    X, y, label_encoders = preparar_datos_regresion(encuestas)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    try:
        joblib.dump(modelo, MODEL_FILE)
        joblib.dump(label_encoders, ENCODERS_FILE)
    except Exception:
        pass  # no crítico si falla guardar

    return modelo, label_encoders

def predecir_por_matricula(matricula):
    # Preparar respuesta base
    try:
        db = conectar_mongodb()
    except Exception as e:
        return {"error": f"Error conectando a MongoDB: {str(e)}"}

    alumno = db.alumnos.find_one({"matricula": matricula})
    if not alumno:
        return {"error": f"No se encontró alumno con matrícula {matricula}"}

    id_alumno = alumno["_id"]
    encuestas = list(db.encuestas.find({"id_alumno": id_alumno}).sort("_id", 1))
    if not encuestas:
        return {"error": f"No se encontraron encuestas para el alumno con matrícula {matricula}"}

    try:
        modelo, label_encoders = cargar_o_entrenar(db)
    except Exception as e:
        return {"error": f"Error entrenando o cargando el modelo: {str(e)}"}

    ultima = encuestas[-1]
    X_pred = preparar_una_encuesta(ultima, label_encoders)

    for col in modelo.feature_names_in_:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[modelo.feature_names_in_]

    try:
        riesgo_pred = modelo.predict(X_pred)[0]
    except Exception as e:
        return {"error": f"Error al predecir: {str(e)}"}

    riesgo = round(riesgo_pred * 100, 2)

    if riesgo >= 80:
        motivo = "Alto riesgo: multiples factores academicos y personales"
        recomendacion = "Asesoria academica urgente y apoyo psicologico"
    elif riesgo >= 60:
        motivo = "Riesgo medio: bajo promedio o problemas personales"
        recomendacion = "Tutoria y monitoreo continuo"
    elif riesgo >= 40:
        motivo = "Riesgo leve: dificultad para estudiar o motivación baja"
        recomendacion = "Seguimiento por tutor y actividades motivacionales"
    else:
        motivo = "Sin riesgo aparente"
        recomendacion = "Mantener seguimiento regular"

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

    resultado = OrderedDict([
        ("matricula", matricula),
        ("nombre_completo", nombre_completo),
        ("nombre_grupo", nombre_grupo),
        ("riesgo", riesgo),
        ("motivo", motivo),
        ("recomendacion", recomendacion),
        ("timestamp", datetime.now(timezone.utc).isoformat())
    ])

    filtro = {"id_alumno": id_alumno}
    update = {
        "$set": {
            "matricula": resultado["matricula"],
            "nombre_completo": resultado["nombre_completo"],
            "nombre_grupo": resultado["nombre_grupo"],
            "riesgo": resultado["riesgo"],
            "motivo": resultado["motivo"],
            "recomendacion": resultado["recomendacion"],
            "timestamp": resultado["timestamp"]
        }
    }
    try:
        db.predicciones.update_one(filtro, update, upsert=True)
    except Exception as e:
        resultado["warning_guardado"] = f"No se pudo guardar prediccion: {str(e)}"

    return resultado

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Falta matricula como argumento"}))
        sys.exit(1)

    matricula = sys.argv[1].strip()
    resultado = predecir_por_matricula(matricula)
    print(json.dumps(resultado, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
