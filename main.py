import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

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

def preparar_datos(encuestas):
    datos = [normalizar_documento(doc) for doc in encuestas]
    df = pd.DataFrame(datos)

    # Eliminar filas con datos nulos en variables clave
    required = [
        "trabaja", "ingreso_mensual", "padecimiento_cronico",
        "atencion_psicologica", "materias_reprobadas", "promedio_previo",
        "motivacion", "dificultad_estudio", "expectativa_terminar", "abandona"
    ]
    df = df.dropna(subset=required)

    # Mapear SÃ­/No a 1/0 para variables binarias
    for col in ["trabaja", "padecimiento_cronico", "atencion_psicologica"]:
        df[col] = df[col].map({"SÃ­": 1, "No": 0}).fillna(0).astype(int)

    # Etiqueta binaria
    y = df["abandona"].map(lambda x: 1 if x == "SÃ­" else 0)

    X = df.drop(columns=["id_alumno", "abandona", "matricula"], errors="ignore")

    # Codificar variables categÃ³ricas y guardar los encoders
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, y, df, label_encoders

def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print("ðŸ“Š EvaluaciÃ³n del modelo:\n", classification_report(y_test, y_pred))

    return modelo

def guardar_modelo(modelo, label_encoders):
    os.makedirs('modelo_entrenado', exist_ok=True)
    joblib.dump(modelo, 'modelo_entrenado/modelo.pkl')
    joblib.dump(label_encoders, 'modelo_entrenado/label_encoders.pkl')
    print("âœ… Modelo y encoders guardados en la carpeta 'modelo_entrenado'")

def cargar_modelo():
    modelo = joblib.load('modelo_entrenado/modelo.pkl')
    label_encoders = joblib.load('modelo_entrenado/label_encoders.pkl')
    return modelo, label_encoders

def predecir_riesgo(modelo, df, datos_originales, label_encoders):
    X_pred = df.drop(columns=["id_alumno", "abandona", "matricula"], errors="ignore")
    
    # Aplicar los mismos label encoders
    for col, le in label_encoders.items():
        if col in X_pred.columns:
            try:
                X_pred[col] = le.transform(X_pred[col].astype(str))
            except:
                X_pred[col] = 0  # Valor por defecto si falla la transformaciÃ³n

    # Asegurar que tenemos todas las columnas que el modelo espera
    for col in modelo.feature_names_in_:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[modelo.feature_names_in_]

    predicciones = modelo.predict_proba(X_pred)[:, 1]  # Probabilidad de clase positiva

    resultados = []
    for i, riesgo in enumerate(predicciones):
        porcentaje = round(riesgo * 100, 2)

        if porcentaje >= 80:
            motivo = "Alto riesgo: mÃºltiples factores acadÃ©micos y personales"
            recomendacion = "AsesorÃ­a acadÃ©mica urgente y apoyo psicolÃ³gico"
        elif porcentaje >= 60:
            motivo = "Riesgo medio: bajo promedio o problemas personales"
            recomendacion = "TutorÃ­a y monitoreo continuo"
        elif porcentaje >= 40:
            motivo = "Riesgo leve: dificultad para estudiar o motivaciÃ³n baja"
            recomendacion = "Seguimiento por tutor y actividades motivacionales"
        else:
            motivo = "Sin riesgo aparente"
            recomendacion = "Mantener seguimiento regular"

        resultados.append({
            "id_alumno": datos_originales[i].get("id_alumno"),
            "riesgo": porcentaje,
            "motivo": motivo,
            "recomendacion": recomendacion
        })

    return resultados
