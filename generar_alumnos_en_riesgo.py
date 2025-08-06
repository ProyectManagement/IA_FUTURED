from faker import Faker
import random
from pymongo import MongoClient, UpdateOne
import uuid
from datetime import datetime, timezone

fake = Faker()

# Conexión MongoDB
client = MongoClient("mongodb+srv://FutuRed:qotG44JpqoexRsjv@cluster0.yf9o1kh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["tu_base_de_datos"]
collection_encuestas = db["encuestas"]
collection_alumnos = db["alumnos"]

# Carreras y grupos
carreras = [
    {"_id": "67d7a402693e07166103f482", "nombre": "Ingeniería en Desarrollo y gestion de Software"},
    {"_id": "67d7a402693e07166103f483", "nombre": "Ingeniería en redes inteligentes y ciberseguridad"}
]

grupos = [
    {"_id": "67d7a45f693e07166103f485", "nombre": "IDGS-81", "id_carrera": "67d7a402693e07166103f482"},
    {"_id": "67d7a460693e07166103f486", "nombre": "IDGS-82", "id_carrera": "67d7a402693e07166103f482"},
    {"_id": "67d7a461693e07166103f487", "nombre": "IDGS-83", "id_carrera": "67d7a402693e07166103f482"},
    {"_id": "67e1f17496c6160377f484b9", "nombre": "IRIC-81", "id_carrera": "67d7a402693e07166103f483"},
    {"_id": "67e1f1ac96c6160377f484bb", "nombre": "IRIC-82", "id_carrera": "67d7a402693e07166103f483"},
    {"_id": "67e1f1e796c6160377f484bc", "nombre": "IRIC-83", "id_carrera": "67d7a402693e07166103f483"},
]

def generar_registro(riesgo_porcentaje):
    id_alumno = str(uuid.uuid4())
    grupo = random.choice(grupos)
    carrera_id = grupo["id_carrera"]
    riesgo_texto = "Sí" if riesgo_porcentaje >= 50 else "No"
    nombre = fake.first_name()
    apellido_paterno = fake.last_name()
    apellido_materno = fake.last_name()
    matricula = str(fake.random_number(digits=11, fix_len=True))
    now = datetime.now(timezone.utc)

    encuesta_doc = {
        "_id": str(uuid.uuid4()),
        "id_alumno": id_alumno,
        "matricula": matricula,
        "nombre": nombre,
        "apellido_paterno": apellido_paterno,
        "apellido_materno": apellido_materno,
        "correo": fake.email(),
        "id_carrera": carrera_id,
        "id_grupo": grupo["_id"],
        "curp": fake.lexify(text='??????????????????'),
        "genero": random.choice(["Hombre", "Mujer", "Otro"]),
        "edad": random.randint(18, 30),
        "telefono_celular": fake.phone_number(),
        "telefono_casa": fake.phone_number(),
        "direccion": {
            "calle": fake.street_name(),
            "no_exterior": str(random.randint(1, 9999)),
            "no_interior": None,
            "colonia": fake.city_suffix(),
            "cp": fake.postcode(),
            "municipio": fake.city()
        },
        "referencias_domicilio": [fake.address(), fake.address()],
        "servicios": {
            "luz": random.choice([True, False]),
            "agua": random.choice([True, False]),
            "internet": random.choice([True, False]),
            "computadora": random.choice([True, False])
        },
        "vivienda": {
            "tipo": random.choice(["Propia", "Rentada", "Familiar"]),
            "monto_renta": random.randint(0, 10000),
            "estado_civil": random.choice(["Soltero", "Casado", "Divorciado"]),
            "numero_hijos": random.randint(0, 5)
        },
        "salud": {
            "embarazada": "No",
            "meses_embarazo": None,
            "atencion_embarazo": None
        },
        "contacto_emergencia_1": {
            "nombre": fake.name(),
            "telefono": fake.phone_number(),
            "relacion": random.choice(["Padre", "Madre", "Hermano", "Tío", "Amigo"])
        },
        "condiciones_salud": {
            "padecimiento_cronico": random.choice(["Sí", "No"]),
            "nombre_padecimiento": None,
            "atencion_psicologica": random.choice(["Sí", "No"]),
            "motivo_atencion": None,
            "horas_sueno": random.choice(["<6", "6-8", ">8"]),
            "alimentacion": random.choice(["Buena", "Regular", "Mala"])
        },
        "aspectos_socioeconomicos": {
            "trabaja": random.choice(["Sí", "No"]),
            "horas_trabajo": None,
            "ingreso_mensual": random.choice([0, 2000, 4000, 6000, 8000]),
            "nombre_trabajo": None,
            "dias_trabajo": None,
            "aporte_familiar": random.choice(["Padre", "Madre", "Hermano", "Otro"]),
            "monto_aporte": random.randint(0, 5000),
            "otro_aporte": None
        },
        "aportantes_gasto_familiar": {
            "ingreso_familiar": random.randint(10000, 50000),
            "gasto_mensual": random.randint(5000, 30000),
            "padre": random.choice([True, False]),
            "madre": random.choice([True, False]),
            "hermanos": random.choice([True, False]),
            "abuelos": random.choice([True, False]),
            "pareja": random.choice([True, False]),
            "otro": random.choice([True, False])
        },
        "analisis_academico": {
            "promedio_previo": round(random.uniform(5.0, 10.0), 2),
            "materias_reprobadas": random.randint(0, 5),
            "repitio_anio": random.choice(["Sí", "No"]),
            "razon_repitio": None,
            "detalle_repitio": None,
            "horas_estudio_diario": random.choice(["<1", "1-2", "2-4", ">4"]),
            "apoyo_academico": random.choice(["Sí", "No"]),
            "tipo_apoyo": None,
            "frecuencia_apoyo": None,
            "motivacion": random.randint(1, 5),
            "dificultad_estudio": random.choice(["Académica", "Salud", "Familiar", "Dinero", "Tiempo"]),
            "expectativa_terminar": random.choice(["Muy seguro", "Seguro", "Poco seguro", "Inseguro"]),
            "comentarios": fake.text(max_nb_chars=200)
        },
        "created_at": now,
        "updated_at": now,
        "abandona": riesgo_texto,
        "riesgo_abandono": riesgo_porcentaje
    }

    alumno_doc = {
        "_id": id_alumno,
        "matricula": matricula,
        "nombre": nombre,
        "apellido_paterno": apellido_paterno,
        "apellido_materno": apellido_materno,
        "id_carrera": carrera_id,
        "id_grupo": grupo["_id"],
        "created_at": now,
        "updated_at": now
    }

    return encuesta_doc, alumno_doc

def insertar_registros():
    alumnos_bulk = []

    for riesgo in range(10, 110, 10):  # 10% al 100%
        for _ in range(100):
            encuesta_doc, alumno_doc = generar_registro(riesgo)

            # Insertar encuesta
            collection_encuestas.insert_one(encuesta_doc)

            # Preparar operación para alumnos
            alumnos_bulk.append(UpdateOne(
                {"_id": alumno_doc["_id"]},
                {"$set": alumno_doc},
                upsert=True
            ))

    print(f"{len(alumnos_bulk)} encuestas insertadas")

    if alumnos_bulk:
        result = collection_alumnos.bulk_write(alumnos_bulk)
        print(f"Alumnos actualizados/insertados: {result.upserted_count + result.modified_count}")

if __name__ == "__main__":
    insertar_registros()
