# main.py - Backend Clasificador de Ganado CON DETECTOR PREVIO
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import uvicorn

# ========== IMPORTAR DETECTOR DE ANIMALES ==========
from ultralytics import YOLO

# ========== MODELOS DE DATOS ==========
class SolicitudClasificacion(BaseModel):
    porcentaje_minimo: float  # Porcentaje mínimo de confianza (ej: 5.0 para 5%)

class ValidacionAnimal(BaseModel):
    """Resultado de la validación animal/vaca"""
    es_animal: bool
    es_vaca: bool
    confianza_animal: float  # Porcentaje de confianza que es animal
    confianza_vaca: float   # Porcentaje de confianza que es vaca
    animales_detectados: List[Dict]  # Lista de animales encontrados

class DescripcionRaza(BaseModel):
    nombre: str
    peso_promedio_kg: str
    altura_promedio_cm: str
    esperanza_vida_anos: str
    origen: str
    produccion_leche_litros_dia: str
    caracteristicas_principales: List[str]
    temperamento: str

class ResultadoClasificacion(BaseModel):
    nombre_raza: str
    confiabilidad: float  # Porcentaje de confianza (ej: 85.3)
    descripcion: Optional[DescripcionRaza] = None  # Solo para el ganador

class RespuestaClasificacionCompleta(BaseModel):
    """Respuesta completa con validación + clasificación"""
    exito: bool
    mensaje: str
    es_vaca: bool  # NUEVO: Resultado directo si es vaca
    porcentaje_vaca: float  # NUEVO: Porcentaje de confianza que es vaca
    validacion_animal: ValidacionAnimal
    clasificaciones: Optional[List[ResultadoClasificacion]] = None  # Solo si es vaca
    fecha_procesamiento: str

class RespuestaAsincrona(BaseModel):
    id_transaccion: str
    estado: str  # "procesando", "completado", "error"
    mensaje: str

# ========== CONFIGURACIÓN DE LA APLICACIÓN ==========
app = FastAPI(
    title="🐄 API Clasificador de Razas de Ganado CON DETECTOR",
    description="Sistema completo: Detecta animales → Verifica que sea vaca → Clasifica raza",
    version="2.0.0"
)

# Configurar CORS para permitir llamadas desde frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== VARIABLES GLOBALES ==========
modelo_ia = None  # Modelo para clasificar razas
modelo_detector = None  # Modelo YOLO para detectar animales
nombres_razas = []
transacciones_pendientes: Dict = {}

# ========== BASE DE DATOS DE RAZAS (igual que antes) ==========
DESCRIPCIONES_RAZAS = {
    "Holstein Friesian cattle": DescripcionRaza(
        nombre="Holstein Friesian",
        peso_promedio_kg="580-680 kg (hembras), 900-1000 kg (machos)",
        altura_promedio_cm="145-150 cm (hembras), 160-170 cm (machos)", 
        esperanza_vida_anos="6-8 años productivos, hasta 20 años total",
        origen="Países Bajos y Norte de Alemania",
        produccion_leche_litros_dia="25-40 litros por día",
        caracteristicas_principales=[
            "Pelaje blanco y negro característico",
            "Mayor productora de leche del mundo",
            "Adaptable a diferentes climas",
            "Excelente conversión alimenticia"
        ],
        temperamento="Dócil y fácil de manejar"
    ),
    
    "Jersey cattle": DescripcionRaza(
        nombre="Jersey",
        peso_promedio_kg="350-450 kg (hembras), 540-680 kg (machos)",
        altura_promedio_cm="120-125 cm (hembras), 135-145 cm (machos)",
        esperanza_vida_anos="7-9 años productivos, hasta 18 años total", 
        origen="Isla de Jersey, Canal de la Mancha",
        produccion_leche_litros_dia="18-25 litros por día",
        caracteristicas_principales=[
            "Leche con alto contenido graso (4.5-5.5%)",
            "Tamaño pequeño pero muy eficiente",
            "Color marrón claro a oscuro",
            "Excelente para sistemas pastoriles"
        ],
        temperamento="Alerta pero generalmente dócil"
    ),
    
    "Ayrshire cattle": DescripcionRaza(
        nombre="Ayrshire", 
        peso_promedio_kg="450-550 kg (hembras), 700-850 kg (machos)",
        altura_promedio_cm="135-140 cm (hembras), 150-160 cm (machos)",
        esperanza_vida_anos="6-8 años productivos, hasta 16 años total",
        origen="Condado de Ayr, Escocia",
        produccion_leche_litros_dia="20-30 litros por día",
        caracteristicas_principales=[
            "Pelaje rojizo y blanco",
            "Muy resistente y adaptable",
            "Buena calidad de leche",
            "Excelente para climas fríos"
        ],
        temperamento="Activo pero manejable"
    ),
    
    "Brown Swiss cattle": DescripcionRaza(
        nombre="Brown Swiss",
        peso_promedio_kg="590-680 kg (hembras), 900-1100 kg (machos)", 
        altura_promedio_cm="142-148 cm (hembras), 155-165 cm (machos)",
        esperanza_vida_anos="7-9 años productivos, hasta 20 años total",
        origen="Suiza",
        produccion_leche_litros_dia="22-32 litros por día",
        caracteristicas_principales=[
            "Color marrón sólido característico",
            "Segunda raza lechera más antigua",
            "Muy longevas y resistentes",
            "Leche ideal para fabricación de quesos"
        ],
        temperamento="Tranquilo y dócil"
    ),
    
    "Red Dane cattle": DescripcionRaza(
        nombre="Red Dane",
        peso_promedio_kg="550-650 kg (hembras), 850-1000 kg (machos)",
        altura_promedio_cm="140-145 cm (hembras), 150-160 cm (machos)", 
        esperanza_vida_anos="6-8 años productivos, hasta 18 años total",
        origen="Dinamarca",
        produccion_leche_litros_dia="20-28 litros por día",
        caracteristicas_principales=[
            "Color rojizo uniforme",
            "Excelente adaptación al clima nórdico", 
            "Buena fertilidad",
            "Resistente a enfermedades"
        ],
        temperamento="Calmado y fácil manejo"
    )
}

# ========== EVENTOS DE INICIO ==========
@app.on_event("startup")
async def cargar_modelos_al_iniciar():
    """Cargar AMBOS modelos: detector + clasificador"""
    global modelo_ia, modelo_detector, nombres_razas
    
    try:
        print("🔄 Cargando modelos de IA...")
        
        # 1. Cargar detector de animales (YOLO)
        print("🔄 Cargando detector de animales...")
        modelo_detector = YOLO('yolov8n.pt')
        print("✅ Detector de animales cargado (YOLO)")
        
        # 2. Cargar clasificador de razas
        print("🔄 Cargando clasificador de razas...")
        modelo_ia = tf.keras.models.load_model('clasificador_ganado_final.keras')
        print("✅ Clasificador de razas cargado")
        
        # 3. Cargar nombres de las razas
        with open('clases_ganado.json', 'r', encoding='utf-8') as archivo:
            nombres_razas = json.load(archivo)
        print(f"✅ Razas cargadas: {nombres_razas}")
        
        # 4. Cargar información adicional del modelo
        with open('info_modelo.json', 'r', encoding='utf-8') as archivo:
            info_modelo = json.load(archivo)
        print(f"✅ Info del modelo: {info_modelo['arquitectura']} - Precisión: {info_modelo['precision_final']:.2%}")
        
        print("🚀 Sistema completo listo: Detector + Clasificador")
        
    except Exception as error:
        print(f"❌ Error al cargar modelos: {error}")
        modelo_ia = None
        modelo_detector = None

# ========== NUEVA FUNCIÓN: DETECTOR DE ANIMALES/VACAS ==========
def detectar_animal_y_vaca(bytes_imagen: bytes) -> ValidacionAnimal:
    """
    PASO 1: Detectar si es animal y si es vaca usando YOLO
    """
    try:
        # Convertir bytes a imagen PIL
        imagen_pil = Image.open(io.BytesIO(bytes_imagen))
        
        # Detectar objetos con YOLO
        results = modelo_detector(imagen_pil)
        
        # Lista de animales que detecta YOLO
        animales_yolo = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                        'elephant', 'bear', 'zebra', 'giraffe']
        
        es_animal = False
        es_vaca = False
        confianza_animal = 0.0
        confianza_vaca = 0.0
        animales_detectados = []
        
        # Procesar resultados
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = modelo_detector.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Verificar si es animal con confianza > 60%
                    if class_name in animales_yolo and confidence > 0.6:
                        es_animal = True
                        confianza_animal = max(confianza_animal, confidence * 100)
                        
                        animales_detectados.append({
                            "animal": class_name,
                            "confianza": round(confidence * 100, 1)
                        })
                        
                        # Verificar específicamente si es vaca
                        if class_name == 'cow':
                            es_vaca = True
                            confianza_vaca = confidence * 100
        
        return ValidacionAnimal(
            es_animal=es_animal,
            es_vaca=es_vaca,
            confianza_animal=round(confianza_animal, 1),
            confianza_vaca=round(confianza_vaca, 1),
            animales_detectados=animales_detectados
        )
        
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error en detección de animales: {str(error)}"
        )

# ========== FUNCIONES AUXILIARES (modificadas) ==========
def preprocesar_imagen_ganado(bytes_imagen: bytes) -> np.ndarray:
    """
    Convertir imagen a formato que entiende el modelo de clasificación de razas
    """
    try:
        # Abrir imagen desde bytes
        imagen_pil = Image.open(io.BytesIO(bytes_imagen))
        
        # Convertir a RGB si es necesario (eliminar canal alpha)
        if imagen_pil.mode != 'RGB':
            imagen_pil = imagen_pil.convert('RGB')
        
        # Redimensionar a 224x224 píxeles (tamaño que espera el modelo)
        imagen_redimensionada = imagen_pil.resize((224, 224))
        
        # Convertir a array numérico
        array_imagen = np.array(imagen_redimensionada)
        
        # Normalizar píxeles de 0-255 a 0-1
        array_normalizado = array_imagen.astype('float32') / 255.0
        
        # Añadir dimensión de batch (el modelo espera lotes de imágenes)
        array_con_batch = np.expand_dims(array_normalizado, axis=0)
        
        return array_con_batch
        
    except Exception as error:
        raise HTTPException(
            status_code=400, 
            detail=f"Error procesando imagen: {str(error)}"
        )

def ejecutar_clasificacion_razas(array_imagen: np.ndarray, porcentaje_minimo: float) -> List[ResultadoClasificacion]:
    """
    PASO 2: Clasificar raza de vaca (solo si ya se confirmó que es vaca)
    """
    # Hacer predicción con el modelo de clasificación de razas
    predicciones_raw = modelo_ia.predict(array_imagen, verbose=0)[0]
    
    # Crear lista de resultados para TODAS las 5 razas
    resultados_completos = []
    
    # Procesar cada raza (siempre las 5)
    for indice, (nombre_raza, probabilidad) in enumerate(zip(nombres_razas, predicciones_raw)):
        # Convertir probabilidad a porcentaje
        porcentaje_confianza = float(probabilidad) * 100
        
        # Crear objeto de resultado básico
        resultado = ResultadoClasificacion(
            nombre_raza=nombre_raza,
            confiabilidad=round(porcentaje_confianza, 1),
            descripcion=None  # Por defecto vacío
        )
        
        resultados_completos.append(resultado)
    
    # Ordenar por confianza (mayor a menor)
    resultados_completos.sort(key=lambda x: x.confiabilidad, reverse=True)
    
    # Añadir descripción SOLO al ganador (el primero después de ordenar)
    if resultados_completos:
        raza_ganadora = resultados_completos[0].nombre_raza
        if raza_ganadora in DESCRIPCIONES_RAZAS:
            resultados_completos[0].descripcion = DESCRIPCIONES_RAZAS[raza_ganadora]
    
    return resultados_completos

# ========== ENDPOINTS MODIFICADOS ==========
@app.get("/")
async def endpoint_bienvenida():
    """Página de inicio de la API"""
    return {
        "mensaje": "🐄 API Clasificador de Ganado CON DETECTOR",
        "version": "2.0.0",
        "estado": "Funcionando correctamente",
        "flujo_procesamiento": [
            "1. Detecta si es animal (YOLO)",
            "2. Verifica si es vaca",
            "3. Clasifica raza (solo si es vaca)",
            "4. Devuelve resultado completo"
        ],
        "endpoints_disponibles": {
            "clasificar_completo": "POST /clasificar/",
            "clasificar_asincrono": "POST /clasificar-asincrono/",
            "consultar_resultado": "GET /resultado/{id_transaccion}",
            "info_sistema": "GET /info/",
            "salud": "GET /salud/"
        }
    }

@app.get("/salud/")
async def verificar_salud_sistema():
    """Verificar que el sistema esté funcionando"""
    return {
        "estado": "saludable" if (modelo_ia is not None and modelo_detector is not None) else "error",
        "detector_cargado": modelo_detector is not None,
        "clasificador_cargado": modelo_ia is not None,
        "razas_disponibles": len(nombres_razas),
        "transacciones_pendientes": len(transacciones_pendientes),
        "fecha_verificacion": datetime.now().isoformat()
    }

@app.post("/clasificar/", response_model=RespuestaClasificacionCompleta)
async def clasificar_ganado_completo(
    archivo_imagen: UploadFile = File(...),
    porcentaje_minimo: float = 5.0
):
    """
    🔍 CLASIFICACIÓN COMPLETA - Detector + Clasificador
    
    FLUJO:
    1. Detecta si es animal usando YOLO
    2. Verifica si específicamente es vaca
    3. Si es vaca → Clasifica la raza
    4. Si no es vaca → Solo devuelve validación
    
    Parámetros:
    - archivo_imagen: Imagen a analizar
    - porcentaje_minimo: Umbral mínimo de confianza
    
    Respuesta:
    - validacion_animal: Resultado del detector
    - clasificaciones: Razas (solo si es vaca)
    """
    
    # Verificar que ambos modelos estén cargados
    if modelo_ia is None or modelo_detector is None:
        raise HTTPException(
            status_code=500, 
            detail="❌ Modelos de IA no disponibles. Contacte al administrador."
        )
    
    # Validar formato de imagen
    extensiones_validas = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    if archivo_imagen.filename:
        extension = '.' + archivo_imagen.filename.lower().split('.')[-1]
        if extension not in extensiones_validas:
            raise HTTPException(
                status_code=400,
                detail="❌ El archivo debe ser una imagen (JPG, PNG, BMP, etc.)"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="❌ Nombre de archivo no válido"
        )
    
    # Validar porcentaje
    if not (0 <= porcentaje_minimo <= 100):
        raise HTTPException(
            status_code=400,
            detail="❌ El porcentaje debe estar entre 0 y 100"
        )
    
    try:
        print(f"🔄 Procesando imagen: {archivo_imagen.filename}")
        
        # Leer bytes de la imagen
        contenido_imagen = await archivo_imagen.read()
        
        # PASO 1: Detectar animal y vaca
        print("🔍 Paso 1: Detectando animal/vaca...")
        validacion = detectar_animal_y_vaca(contenido_imagen)
        
        clasificaciones = None
        mensaje = ""
        
        if not validacion.es_animal:
            mensaje = "❌ No se detectó ningún animal en la imagen"
            
        elif not validacion.es_vaca:
            animales_str = ", ".join([a["animal"] for a in validacion.animales_detectados])
            mensaje = f"✅ Es un animal ({animales_str}) pero NO es una vaca"
            
        else:
            # PASO 2: Es vaca → Clasificar raza
            print("🔍 Paso 2: Clasificando raza de vaca...")
            
            # Preprocesar imagen para el clasificador de razas
            imagen_procesada = preprocesar_imagen_ganado(contenido_imagen)
            
            # Ejecutar clasificación de razas
            clasificaciones = ejecutar_clasificacion_razas(imagen_procesada, porcentaje_minimo)
            
            mensaje = f"✅ Es una vaca! Raza identificada: {clasificaciones[0].nombre_raza} ({clasificaciones[0].confiabilidad}%)"
        
        print(f"✅ Procesamiento completado: {mensaje}")
        
        # Crear respuesta completa
        respuesta = RespuestaClasificacionCompleta(
            exito=True,
            mensaje=mensaje,
            es_vaca=validacion.es_vaca,  # NUEVO: Resultado directo
            porcentaje_vaca=validacion.confianza_vaca,  # NUEVO: Porcentaje directo
            validacion_animal=validacion,
            clasificaciones=clasificaciones,
            fecha_procesamiento=datetime.now().isoformat()
        )
        
        return respuesta
        
    except Exception as error:
        print(f"❌ Error en clasificación completa: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la clasificación: {str(error)}"
        )

# ========== RESTO DE ENDPOINTS (mantienes los asíncronos igual) ==========
@app.post("/clasificar-asincrono/", response_model=RespuestaAsincrona)
async def clasificar_ganado_asincrono(
    tareas_segundo_plano: BackgroundTasks,
    archivo_imagen: UploadFile = File(...),
    porcentaje_minimo: float = 5.0
):
    """Versión asíncrona de la clasificación completa"""
    # Verificar modelos
    if modelo_ia is None or modelo_detector is None:
        raise HTTPException(status_code=500, detail="❌ Modelos no disponibles")
    
    # Validaciones (igual que antes)
    extensiones_validas = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    if archivo_imagen.filename:
        extension = '.' + archivo_imagen.filename.lower().split('.')[-1]
        if extension not in extensiones_validas:
            raise HTTPException(status_code=400, detail="❌ Debe ser una imagen")
    
    if not (0 <= porcentaje_minimo <= 100):
        raise HTTPException(status_code=400, detail="❌ Porcentaje inválido")
    
    # Generar ID único de transacción
    id_transaccion = str(uuid.uuid4())
    
    # Guardar estado inicial
    transacciones_pendientes[id_transaccion] = {
        "estado": "procesando",
        "fecha_inicio": datetime.now().isoformat(),
        "resultado": None,
        "error": None
    }
    
    # Leer imagen
    contenido_imagen = await archivo_imagen.read()
    
    # Programar procesamiento en segundo plano
    tareas_segundo_plano.add_task(
        procesar_imagen_completa_segundo_plano,
        id_transaccion,
        contenido_imagen,
        porcentaje_minimo
    )
    
    return RespuestaAsincrona(
        id_transaccion=id_transaccion,
        estado="procesando",
        mensaje="Procesamiento completo iniciado (detector + clasificador)"
    )

async def procesar_imagen_completa_segundo_plano(
    id_transaccion: str, 
    contenido_imagen: bytes, 
    porcentaje_minimo: float
):
    """Procesamiento completo en segundo plano"""
    try:
        # Simular tiempo de procesamiento
        await asyncio.sleep(2)
        
        # PASO 1: Detectar animal/vaca
        validacion = detectar_animal_y_vaca(contenido_imagen)
        
        clasificaciones = None
        mensaje = ""
        
        if not validacion.es_animal:
            mensaje = "❌ No se detectó ningún animal"
        elif not validacion.es_vaca:
            animales_str = ", ".join([a["animal"] for a in validacion.animales_detectados])
            mensaje = f"✅ Es un animal ({animales_str}) pero NO es vaca"
        else:
            # PASO 2: Clasificar raza
            imagen_procesada = preprocesar_imagen_ganado(contenido_imagen)
            clasificaciones = ejecutar_clasificacion_razas(imagen_procesada, porcentaje_minimo)
            mensaje = f"✅ Es vaca! Raza: {clasificaciones[0].nombre_raza}"
        
        # Crear resultado
        resultado = RespuestaClasificacionCompleta(
            exito=True,
            mensaje=mensaje,
            es_vaca=validacion.es_vaca,  # NUEVO: Resultado directo
            porcentaje_vaca=validacion.confianza_vaca,  # NUEVO: Porcentaje directo
            validacion_animal=validacion,
            clasificaciones=clasificaciones,
            fecha_procesamiento=datetime.now().isoformat()
        )
        
        # Actualizar estado
        transacciones_pendientes[id_transaccion].update({
            "estado": "completado",
            "resultado": resultado.dict(),
            "fecha_completado": datetime.now().isoformat()
        })
        
    except Exception as error:
        transacciones_pendientes[id_transaccion].update({
            "estado": "error",
            "error": str(error),
            "fecha_error": datetime.now().isoformat()
        })

@app.get("/resultado/{id_transaccion}")
async def consultar_resultado_asincrono(id_transaccion: str):
    """Consultar resultado asíncrono"""
    if id_transaccion not in transacciones_pendientes:
        raise HTTPException(status_code=404, detail="❌ ID no encontrado")
    
    transaccion = transacciones_pendientes[id_transaccion]
    estado = transaccion["estado"]
    
    if estado == "procesando":
        return {
            "id_transaccion": id_transaccion,
            "estado": "procesando", 
            "mensaje": "🔄 Procesamiento en curso (detector + clasificador)...",
            "fecha_inicio": transaccion["fecha_inicio"]
        }
    
    elif estado == "completado":
        resultado_final = transaccion["resultado"]
        # NO eliminar inmediatamente - mantener por si se consulta de nuevo
        print(resultado_final) 
        return {
            "id_transaccion": id_transaccion,
            "estado": "completado",
            "mensaje": "✅ Procesamiento completo terminado",
            "resultado": resultado_final
        }
    
    elif estado == "error":
        error_detalle = transaccion["error"]
        # NO eliminar inmediatamente - mantener por si se consulta de nuevo
        raise HTTPException(status_code=500, detail=f"❌ Error: {error_detalle}")

@app.delete("/limpiar-transacciones/")
async def limpiar_transacciones_completadas():
    """Limpiar transacciones completadas o con error (mantenimiento)"""
    eliminadas = 0
    ids_a_eliminar = []
    
    for id_trans, data in transacciones_pendientes.items():
        if data["estado"] in ["completado", "error"]:
            ids_a_eliminar.append(id_trans)
    
    for id_trans in ids_a_eliminar:
        del transacciones_pendientes[id_trans]
        eliminadas += 1
    
    return {
        "mensaje": f"✅ {eliminadas} transacciones eliminadas",
        "transacciones_restantes": len(transacciones_pendientes)
    }

@app.get("/info/")
async def obtener_info_sistema():
    """Información del sistema completo"""
    return {
        "nombre_sistema": "Clasificador de Ganado con Detector",
        "version": "2.0.0",
        "flujo_procesamiento": [
            "1. Detector YOLO verifica si es animal",
            "2. Detector YOLO verifica si es vaca", 
            "3. Clasificador de razas (solo si es vaca)",
            "4. Respuesta completa con validación + clasificación"
        ],
        "detector_animales": "YOLOv8 (15+ especies animales)",
        "clasificador_razas": "MobileNetV2 + Transfer Learning",
        "razas_soportadas": nombres_razas,
        "total_razas": len(nombres_razas),
        "tamano_imagen_requerido": "224x224 píxeles (clasificador)",
        "formatos_soportados": ["JPG", "JPEG", "PNG", "BMP", "GIF", "WEBP"]
    }

# ========== EJECUTAR SERVIDOR ==========
if __name__ == "__main__":
    print("🐄 Iniciando API Clasificador de Ganado COMPLETO...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )