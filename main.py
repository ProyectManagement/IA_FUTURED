from conexion import conectar_mongodb
from modelo import (entrenar_modelo, preparar_datos, 
                   guardar_modelo, predecir_riesgo)
from pymongo import UpdateOne
from bson import ObjectId

def main():
    # Conectar a MongoDB
    db = conectar_mongodb()
    
    # Obtener y preparar datos
    encuestas = list(db.encuestas.find())
    print(f"Total encuestas encontradas: {len(encuestas)}")
    
    X, y, df, label_encoders = preparar_datos(encuestas)
    print(f"Total datos después de limpieza: {len(X)}")
    
    # Entrenar y guardar modelo
    modelo = entrenar_modelo(X, y)
    guardar_modelo(modelo, label_encoders)
    
    # Predecir riesgo
    resultados = predecir_riesgo(modelo, df, encuestas, label_encoders)
    
    # Actualizar predicciones en MongoDB
    operaciones = []
    for r in resultados:
        id_alumno = r["id_alumno"]
        
        # Buscar datos del alumno
        alumno = db.alumnos.find_one({"_id": id_alumno})
        if alumno:
            nombre_completo = f"{alumno.get('nombre', '')} {alumno.get('apellido_paterno', '')} {alumno.get('apellido_materno', '')}".strip()
            matricula = alumno.get("matricula", "")
            
            grupo_id = alumno.get("id_grupo")
            try:
                grupo = db.grupos.find_one({"_id": ObjectId(grupo_id)})
                nombre_grupo = grupo["nombre"] if grupo else "Desconocido"
            except Exception:
                nombre_grupo = "Desconocido"
        else:
            nombre_completo = "Desconocido"
            matricula = ""
            nombre_grupo = "Desconocido"
        
        # Preparar operación de actualización
        filtro = {"id_alumno": id_alumno}
        update = {
            "$set": {
                "riesgo": r["riesgo"],
                "motivo": r["motivo"],
                "recomendacion": r["recomendacion"],
                "nombre_completo": nombre_completo,
                "matricula": matricula,
                "nombre_grupo": nombre_grupo
            }
        }
        operaciones.append(UpdateOne(filtro, update, upsert=True))
        
        print(f"{matricula} - {nombre_completo} ({nombre_grupo}) → Riesgo: {r['riesgo']}% Motivo: {r['motivo']}")

    # Ejecutar operaciones masivas
    if operaciones:
        result = db.predicciones.bulk_write(operaciones)
        print(f"✅ Predicciones guardadas/actualizadas: {result.modified_count + result.upserted_count}")

if __name__ == "__main__":
    main()