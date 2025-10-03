from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import joblib
import traceback
import pandas as pd

# Intentar importar conexion.py si existe
try:
    from conexion import conectar_mongodb
    _HAS_CONEXION = True
except Exception:
    _HAS_CONEXION = False

def _connect_mongo_from_env():
    from pymongo import MongoClient
    uri = os.getenv('MONGODB_URI', 'mongodb+srv://FutuRed:qotG44JpqoexRsjv@cluster0.yf9o1kh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
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
            _feature_names = list(_model.feature_names_in)
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
    # Solo las features que el modelo conoce
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

    # Map Sí/No a 1/0
    for col in ["trabaja", "padecimiento_cronico", "atencion_psicologica"]:
        if col in df.columns:
            df[col] = df[col].map({"Sí": 1, "Si": 1, "No": 0}).fillna(0).astype(int)

    # Aplicar label encoders
    if _label_encoders:
        for col, le in _label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except Exception:
                    df[col] = 0

    # Mantener solo features del modelo
    if _feature_names:
        df = df.reindex(columns=_feature_names, fill_value=0)

    return df

@app.post("/predict", response_model=PredictResult)
def predict_single(encuesta: EncuestaInput = Body(...)):
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

        return PredictResult(
            id_alumno=doc.get('id_alumno'),
            riesgo=porcentaje,
            motivo=motivo,
            recomendacion=recomendacion
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/from_db', response_model=PredictResult)
def predict_from_db(payload: Dict[str, Any] = Body(...)):
    if 'id_alumno' not in payload:
        raise HTTPException(status_code=400, detail='Falta id_alumno')
    try:
        db = conectar_mongodb() if _HAS_CONEXION else _connect_mongo_from_env()
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
    if 'matricula' not in payload:
        raise HTTPException(status_code=400, detail='Falta matricula')
    try:
        db = conectar_mongodb() if _HAS_CONEXION else _connect_mongo_from_env()
        alumno = db.alumnos.find_one({'matricula': payload['matricula']})
        if not alumno:
            raise HTTPException(status_code=404, detail='Alumno no encontrado')
        enc = db.encuestas.find_one({'id_alumno': str(alumno.get('_id'))})
        if not enc:
            raise HTTPException(status_code=404, detail='Encuesta no encontrada para el alumno')
        enc.setdefault('id_alumno', str(alumno.get('_id')))
        enc.setdefault('matricula', alumno.get('matricula'))
        return predict_single(EncuestaInput(**enc))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
