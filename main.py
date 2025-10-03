"""
FastAPI service for FuturEd prediction model.

Features:
- Loads trained model and label encoders from modelo_entrenado/
- Connects to MongoDB (uses conexion.conectar_mongodb() if available, otherwise MONGODB_URI env)
- Endpoints:
  - GET /health
  - POST /predict (body: encuesta-like JSON) -> returns riesgo, motivo, recomendacion
  - POST /predict/from_db (body: {"id_alumno": "..."}) -> fetches encuesta from Mongo and predicts
  - POST /predict/by_matricula -> fetches alumno by matricula and predicts

Run: uvicorn main:app --reload --port 8000
"""

import os
import traceback
import joblib
import pandas as pd
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------
# MongoDB Connection
# -----------------------------
try:
    from conexion import conectar_mongodb
    _HAS_CONEXION = True
except ImportError:
    _HAS_CONEXION = False

def _connect_mongo_from_env():
    """Fallback MongoDB connection from environment variables"""
    from pymongo import MongoClient
    uri = os.getenv(
        'MONGODB_URI',
        'mongodb+srv://FutuRed:qotG44JpqoexRsjv@cluster0.yf9o1kh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    )
    client = MongoClient(uri)
    dbname = os.getenv('MONGODB_DB', 'futured')
    return client[dbname]

# -----------------------------
# Paths and Global Variables
# -----------------------------
MODEL_PATH = os.getenv('MODEL_PATH', 'modelo_entrenado/modelo.pkl')
ENCODERS_PATH = os.getenv('ENCODERS_PATH', 'modelo_entrenado/label_encoders.pkl')

_model = None
_label_encoders = None
_feature_names = None

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="FuturEd - IA de predicción de abandono")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Schemas
# -----------------------------
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
    matricula: Optional[str]
    nombre_completo: Optional[str]
    nombre_grupo: Optional[str]
    riesgo: float
    motivo: str
    recomendacion: str

# -----------------------------
# Startup: Load Model & Encoders
# -----------------------------
@app.on_event("startup")
def load_resources():
    global _model, _label_encoders, _feature_names
    try:
        _model = joblib.load(MODEL_PATH)
        _label_encoders = joblib.load(ENCODERS_PATH)
        try:
            _feature_names = list(_model.feature_names_in)
        except Exception:
            _feature_names = None
        print(f"Modelo cargado desde: {MODEL_PATH}")
    except Exception as e:
        print("No se pudo cargar el modelo en startup:", str(e))
        _model = None
        _label_encoders = None

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

# -----------------------------
# Utils
# -----------------------------
def _normalize_document_for_model(doc: Dict[str, Any]) -> Dict[str, Any]:
    aspectos = doc.get('aspectos_socioeconomicos', {})
    salud = doc.get('condiciones_salud', {})
    academico = doc.get('analisis_academico', {})

    return {
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

    # Map Sí/No to 1/0
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

    # Keep only features used in training
    if _feature_names:
        df = df.reindex(columns=_feature_names, fill_value=0)

    return df

def _predict_riesgo(X: pd.DataFrame) -> (float, str, str):
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

    return porcentaje, motivo, recomendacion

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict", response_model=PredictResult)
def predict_single(encuesta: EncuestaInput = Body(...)):
    try:
        doc = encuesta.dict()
        X = _prepare_X_from_document(doc)
        riesgo, motivo, recomendacion = _predict_riesgo(X)

        return PredictResult(
            id_alumno=doc.get('id_alumno'),
            matricula=doc.get('matricula'),
            nombre_completo=doc.get('nombre_completo'),
            nombre_grupo=doc.get('nombre_grupo'),
            riesgo=riesgo,
            motivo=motivo,
            recomendacion=recomendacion
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/by_matricula', response_model=PredictResult)
def predict_by_matricula(payload: Dict[str, Any] = Body(...)):
    if 'matricula' not in payload:
        raise HTTPException(status_code=400, detail='Falta matricula')

    try:
        db = conectar_mongodb() if _HAS_CONEXION else _connect_mongo_from_env()

        # Fetch alumno
        alumno = db.alumnos.find_one({'matricula': payload['matricula']})
        if not alumno:
            raise HTTPException(status_code=404, detail='Alumno no encontrado')

        # Fetch encuesta
        enc = db.encuestas.find_one({'id_alumno': str(alumno.get('_id'))})
        if not enc:
            raise HTTPException(status_code=404, detail='Encuesta no encontrada para el alumno')

        # Inject additional info
        enc['id_alumno'] = str(alumno.get('_id'))
        enc['matricula'] = alumno.get('matricula')
        enc['nombre_completo'] = f"{alumno.get('nombre', '')} {alumno.get('app', '')} {alumno.get('apm', '')}".strip()
        enc['nombre_grupo'] = db.grupo.find_one({'_id': enc.get('id_grupo')}).get('nombre') if enc.get('id_grupo') else None

        return predict_single(EncuestaInput(**enc))

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Run app directly
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        reload=True
    )
