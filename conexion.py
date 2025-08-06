# conexion.py
from pymongo import MongoClient

def conectar_mongodb():
    uri = "mongodb+srv://FutuRed:qotG44JpqoexRsjv@cluster0.yf9o1kh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    cliente = MongoClient(uri)
    db = cliente["tu_base_de_datos"]
    return db
