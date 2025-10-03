"""
FastAPI service for FuturEd prediction model.
- Loads trained model and label encoders from modelo_entrenado/
- Connects to MongoDB (uses conexion.conectar_mongodb() if available, otherwise MONGODB_URI env)
- Endpoints:
  - GET /health
  - POST /predict (body: encuesta-like JSON) -> returns riesgo, motivo, recomendacion
  - POST /predict/from_db (body: {"id_alumno": "..."}) -> fetches encuesta from Mongo and predicts
  - POST /predict/batch -> predicts for all encuestas and optionally saves to predicciones collection

Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from bson import ObjectId
import os
import joblib
import pandas as pd
import traceback

# Optional import for custom MongoDB connection
try:
    from conexion import conectar_mongodb
    _HAS_CONEXION = True
except Exception:
    _HAS_CONEXION = False

def _connect_mongo_from_env():
    from pymongo import MongoClient
    uri = os.getenv(
        'MONGODB_URI',
        'mongodb+srv://FutuRed:qotG44JpqoexRsjv@cluster0.yf9o1kh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    )
    client = MongoClient(uri)
    dbname = os.getenv('MONGODB_DB', 'futured')
    return client[dbname]

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

    class Config:
        extra = 'allow'

class PredictResult(BaseModel):
    id_alumno: Optional[str]
    nombre: Optional[str]
    riesgo: float
    motivo: str
    recomendacion: str

@app.on_event("startup")
def load_resources():
    global _model, _label_encoders, _feature_names
    try:
        _model = joblib.load(MODEL_PATH)
        _label_encoders = joblib.load(ENCODERS_PATH)
        try:
            _feature_names = list(_model.feature_names_in_)
        except:
            _feature_names = None
        print(f"✅ Modelo cargado desde: {MODEL_PATH}")
    except Exception as e:
        print("❌ No se pudo cargar el modelo en startup:", str(e))
        _model = None
        _label_encoders = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

def _normalize_document_for_model(doc: Dict[str, Any]) -> Dict[str, Any]:
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

def _prepare_X_from_document(doc: Dict[str, Any]) -> pd.DataFrame:
    if _model is None:
        raise RuntimeError("Modelo no cargado")

    normalized = _normalize_document_for_model(doc)
    df = pd.DataFrame([normalized])

    for col in ["trabaja", "padecimiento_cronico", "atencion_psicologica"]:
        if col in df.columns:
            df[col] = df[col].map({"Sí": 1, "Si": 1, "No": 0}).fillna(0).astype(int)

    if _label_encoders:
        for col, le in _label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except:
                    df[col] = 0

    if _feature_names:
        for col in _feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[_feature_names]

    return df

def _compute_prediction(doc: Dict[str, Any]) -> Dict[str, Any]:
    X = _prepare_X_from_document(doc)
    prob = float(_model.predict_proba(X)[:, 1][0])
    porcentaje = round(prob * 100, 2)

    if porcentaje >= 80:
        motivo = "Alto riesgo: múltiples factores académicos y personales"
        recomendacion = "Intervención inmediata con tutor y apoyo psicológico"
    elif porcentaje >= 50:
        motivo = "Riesgo moderado: indicadores mixtos"
        recomendacion = "Seguimiento académico y apoyo adicional"
    else:
        motivo = "Bajo riesgo: condiciones favorables"
        recomendacion = "Continuar con seguimiento regular"

    return {
        "riesgo": porcentaje,
        "motivo": motivo,
        "recomendacion": recomendacion
    }

@app.post("/predict", response_model=PredictResult)
def predict_one(encuesta: EncuestaInput):
    try:
        doc = encuesta.dict()
        pred = _compute_prediction(doc)
        return {
            "id_alumno": doc.get("id_alumno"),
            "nombre": doc.get("nombre"),
            "riesgo": pred["riesgo"],
            "motivo": pred["motivo"],
            "recomendacion": pred["recomendacion"],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/from_db", response_model=PredictResult)
def predict_from_db(payload: Dict[str, Any] = Body(...)):
    id_alumno = payload.get("id_alumno")
    if not id_alumno:
        raise HTTPException(status_code=400, detail="Falta id_alumno en el cuerpo")

    try:
        db = conectar_mongodb() if _HAS_CONEXION else _connect_mongo_from_env()
        encuestas = db["encuestas"]
        doc = encuestas.find_one({"id_alumno": id_alumno})
        if not doc:
            raise HTTPException(status_code=404, detail="No se encontró la encuesta")

        pred = _compute_prediction(doc)
        return {
            "id_alumno": doc.get("id_alumno"),
            "nombre": doc.get("nombre"),
            "riesgo": pred["riesgo"],
            "motivo": pred["motivo"],
            "recomendacion": pred["recomendacion"],
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch():
    try:
        db = conectar_mongodb() if _HAS_CONEXION else _connect_mongo_from_env()
        encuestas_col = db["encuestas"]
        docs = list(encuestas_col.find({}))

        results = []
        for d in docs:
            try:
                pred = _compute_prediction(d)
                results.append({
                    "id_alumno": d.get("id_alumno"),
                    "nombre": d.get("nombre"),
                    "riesgo": pred["riesgo"],
                    "motivo": pred["motivo"],
                    "recomendacion": pred["recomendacion"],
                })
            except Exception:
                traceback.print_exc()

        return {"cantidad": len(results), "resultados": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
