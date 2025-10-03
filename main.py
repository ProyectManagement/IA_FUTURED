"""
FastAPI service for FuturEd prediction model.
- Loads trained model and label encoders from `modelo_entrenado/`
- Connects to MongoDB (uses `conexion.conectar_mongodb()` if available, otherwise MONGODB_URI env)
- Endpoints:
  - GET /health
  - POST /predict (body: encuesta-like JSON) -> returns riesgo, motivo, recomendacion
  - POST /predict/from_db (body: {"id_alumno": "..."}) -> fetches encuesta from Mongo and predicts
  - POST /predict/batch -> predicts for all encuestas and optionally saves to `predicciones` collection

Run: `uvicorn main:app --reload --port 8000`
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
import os
import joblib
import traceback

# Optional import: use existing conexion.py if user has it
try:
    from conexion import conectar_mongodb
    _HAS_CONEXION = True
except Exception:
    _HAS_CONEXION = False

# Utility: connect to MongoDB if conexion not available
def _connect_mongo_from_env():
    from pymongo import MongoClient
    uri = os.getenv('MONGODB_URI', 'mongodb+srv://FutuRed:qotG44JpqoexRsjv@cluster0.yf9o1kh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    client = MongoClient(uri)
    # If the URI contains a database name after the slash, pymongo's ``MongoClient(uri)``
    # still gives access to client.get_default_database(), but for simplicity we'll use 'futured'
    dbname = os.getenv('MONGODB_DB', 'futured')
    return client[dbname]

# Load model + encoders on startup
MODEL_PATH = os.getenv('MODEL_PATH', 'modelo_entrenado/modelo.pkl')
ENCODERS_PATH = os.getenv('ENCODERS_PATH', 'modelo_entrenado/label_encoders.pkl')

_model = None
_label_encoders = None
_feature_names = None

app = FastAPI(title="FuturEd - IA de predicción de abandono")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EncuestaInput(BaseModel):
    id_alumno: Optional[str]
    matricula: Optional[str]
    aspectos_socioeconomicos: Optional[Dict[str, Any]] = None
    condiciones_salud: Optional[Dict[str, Any]] = None
    analisis_academico: Optional[Dict[str, Any]] = None
    # Accept arbitrary extra fields too
    class Config:
        extra = 'allow'

class PredictResult(BaseModel):
    id_alumno: Optional[str]
    riesgo: float
    motivo: str
    recomendacion: str

@app.on_event("startup")
def load_resources():
    global _model, _label_encoders, _feature_names
    try:
        _model = joblib.load(MODEL_PATH)
        _label_encoders = joblib.load(ENCODERS_PATH)
        # feature names available in scikit-learn >= 1.0 via attribute
        try:
            _feature_names = list(_model.feature_names_in_)
        except Exception:
            _feature_names = None
        print(f"Modelo cargado desde: {MODEL_PATH}")
    except Exception as e:
        print("No se pudo cargar el modelo en startup:", str(e))
        _model = None
        _label_encoders = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


def _normalize_document_for_model(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Keep consistent with modelo.normalizar_documento
    aspectos = doc.get('aspectos_socioeconomicos', {})
    salud = doc.get('condiciones_salud', {})
    academico = doc.get('analisis_academico', {})

    return {
        "id_alumno": doc.get('id_alumno'),
        "matricula": doc.get('matricula'),
        "trabaja": aspectos.get('trabaja'),
        "ingreso_mensual": aspectos.get('ingreso_mensual'),
        "padecimiento_cronico": salud.get('padecimiento_cronico'),
        "atencion_psicologica": salud.get('atencion_psicologica'),
        "horas_sueno": salud.get('horas_sueno'),
        "alimentacion": salud.get('alimentacion'),
        "materias_reprobadas": academico.get('materias_reprobadas'),
        "promedio_previo": academico.get('promedio_previo'),
        "motivacion": academico.get('motivacion'),
        "dificultad_estudio": academico.get('dificultad_estudio'),
        "expectativa_terminar": academico.get('expectativa_terminar')
    }

import pandas as pd

def _prepare_X_from_document(doc: Dict[str, Any]) -> pd.DataFrame:
    if _model is None:
        raise RuntimeError("Modelo no cargado")

    normalized = _normalize_document_for_model(doc)
    df = pd.DataFrame([normalized])

    # Map Sí/No to 1/0 for specific columns if present
    for col in ["trabaja", "padecimiento_cronico", "atencion_psicologica"]:
        if col in df.columns:
            df[col] = df[col].map({"Sí": 1, "Si": 1, "No": 0}).fillna(0).astype(int)

    # Apply label encoders
    if _label_encoders:
        for col, le in _label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except Exception:
                    df[col] = 0

    # Ensure all model features exist
    if _feature_names:
        for col in _feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[_feature_names]

    return df

@app.post("/predict", response_model=PredictResult)
def predict_single(encuesta: EncuestaInput = Body(...)):
    """Recibe una encuesta en el cuerpo y devuelve la probabilidad de abandono con motivo y recomendacion."""
    try:
        doc = encuesta.dict()
        X = _prepare_X_from_document(doc)
        prob = float(_model.predict_proba(X)[:, 1][0])
        porcentaje = round(prob * 100, 2)

        if porcentaje >= 80:
            motivo = "Alto riesgo: múltiples factores académicos y personales"
            recomendacion = "Asesoría académica urgente y apoyo psicológico"
        elif porcentaje >= 60:
            motivo = "Riesgo medio: bajo promedio o problemas personales"
            recomendacion = "Tutoría y monitoreo continuo"
        elif porcentaje >= 40:
            motivo = "Riesgo leve: dificultad para estudiar o motivación baja"
            recomendacion = "Seguimiento por tutor y actividades motivacionales"
        else:
            motivo = "Sin riesgo aparente"
            recomendacion = "Mantener seguimiento regular"

        return PredictResult(id_alumno=doc.get('id_alumno'), riesgo=porcentaje, motivo=motivo, recomendacion=recomendacion)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/from_db', response_model=PredictResult)
def predict_from_db(payload: Dict[str, Any] = Body(...)):
    """Recibe {'id_alumno': '...'} y busca la encuesta correspondiente en MongoDB para predecir."""
    if 'id_alumno' not in payload:
        raise HTTPException(status_code=400, detail='Falta id_alumno')

    try:
        if _HAS_CONEXION:
            db = conectar_mongodb()
        else:
            db = _connect_mongo_from_env()

        enc = db.encuestas.find_one({'id_alumno': payload['id_alumno']})
        if not enc:
            raise HTTPException(status_code=404, detail='Encuesta no encontrada')

        return predict_single(EncuestaInput(**enc))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/by_matricula', response_model=PredictResult)
def predict_by_matricula(payload: Dict[str, Any] = Body(...)):
    """Recibe {"matricula": "..."} y busca al alumno y su encuesta para predecir."""
    if 'matricula' not in payload:
        raise HTTPException(status_code=400, detail='Falta matricula')

    try:
        if _HAS_CONEXION:
            db = conectar_mongodb()
        else:
            db = _connect_mongo_from_env()

        alumno = db.alumnos.find_one({'matricula': payload['matricula']})
        if not alumno:
            raise HTTPException(status_code=404, detail='Alumno no encontrado')

        enc = db.encuestas.find_one({'id_alumno': str(alumno.get('_id'))})
        if not enc:
            raise HTTPException(status_code=404, detail='Encuesta no encontrada para el alumno')

        # Inyectar id_alumno y matricula en el documento por si faltan
        enc.setdefault('id_alumno', str(alumno.get('_id')))
        enc.setdefault('matricula', alumno.get('matricula'))
        return predict_single(EncuestaInput(**enc))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/batch')
def predict_batch(save: bool = True):
    """Predice para todas las encuestas existentes. Si save=True, guarda/actualiza en colección 'predicciones'.
    Retorna un resumen con contadores.
    """
    try:
        if _HAS_CONEXION:
            db = conectar_mongodb()
        else:
            db = _connect_mongo_from_env()

        encuestas = list(db.encuestas.find())
        results = []
        updates = []
        for enc in encuestas:
            try:
                res = predict_single(EncuestaInput(**enc))
                results.append(res.dict())
                if save:
                    filtro = {'id_alumno': res.id_alumno}
                    doc = {
                        'id_alumno': res.id_alumno,
                        'riesgo': res.riesgo,
                        'motivo': res.motivo,
                        'recomendacion': res.recomendacion
                    }
                    updates.append(doc)
            except Exception:
                # skip failing records but continue
                traceback.print_exc()
                continue

        if save and updates:
            from pymongo import UpdateOne
            operations = [UpdateOne({'id_alumno': u['id_alumno']}, {'$set': u}, upsert=True) for u in updates]
            result = db.predicciones.bulk_write(operations)
            return {"predicciones_procesadas": len(updates), "modified": result.modified_count, "upserted": len(result.upserted_ids)}

        return {"predicciones_procesadas": len(results)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# If the user runs this script directly, start uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=int(os.getenv('PORT', 8000)), reload=True)

