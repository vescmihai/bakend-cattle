# debug_backend.py - Diagnosticar problemas del backend
import requests
import json
from pathlib import Path

URL_BASE = "http://localhost:8000"
CARPETA_FOTOS = Path("fotos de vacas")

def verificar_archivos_backend():
    """
    🔍 Verificar si existen los archivos necesarios del modelo
    """
    print("=" * 60)
    print("🔍 VERIFICANDO ARCHIVOS DEL BACKEND")
    print("=" * 60)
    
    archivos_necesarios = [
        "clasificador_ganado_final.keras",
        "clases_ganado.json", 
        "info_modelo.json"
    ]
    
    ruta_backend = Path(".")  # Carpeta actual donde está main.py
    
    print(f"📁 Verificando en: {ruta_backend.absolute()}")
    print()
    
    for archivo in archivos_necesarios:
        ruta_archivo = ruta_backend / archivo
        if ruta_archivo.exists():
            tamaño = ruta_archivo.stat().st_size / (1024*1024)  # MB
            print(f"✅ {archivo:<35} ({tamaño:.1f} MB)")
        else:
            print(f"❌ {archivo:<35} NO ENCONTRADO")
    
    print()
    
    # Verificar contenido de clases_ganado.json si existe
    try:
        with open("clases_ganado.json", 'r', encoding='utf-8') as f:
            clases = json.load(f)
        print(f"📋 Clases en clases_ganado.json: {clases}")
    except:
        print("❌ No se pudo leer clases_ganado.json")
    
    print()

def probar_imagen_simple():
    """
    🧪 Probar con la imagen más simple posible
    """
    print("=" * 60)
    print("🧪 PRUEBA CON IMAGEN SIMPLE")
    print("=" * 60)
    
    # Usar la primera imagen disponible
    archivos_jpg = list(CARPETA_FOTOS.glob("*.jpg"))
    if not archivos_jpg:
        print("❌ No hay imágenes disponibles")
        return
    
    imagen_prueba = archivos_jpg[0]
    print(f"📤 Probando con: {imagen_prueba.name}")
    print(f"📏 Tamaño: {imagen_prueba.stat().st_size / 1024:.1f} KB")
    
    try:
        with open(imagen_prueba, 'rb') as archivo:
            archivos = {'archivo_imagen': archivo}
            datos = {'porcentaje_minimo': 5.0}
            
            print("⏳ Enviando...")
            respuesta = requests.post(
                f"{URL_BASE}/clasificar/",
                files=archivos,
                data=datos,
                timeout=30  # 30 segundos de timeout
            )
        
        print(f"📊 Código de respuesta: {respuesta.status_code}")
        
        if respuesta.status_code == 200:
            resultado = respuesta.json()
            print("✅ ¡FUNCIONA!")
            print(f"🏆 Ganador: {resultado['clasificaciones'][0]['nombre_raza']}")
            print(f"📈 Confianza: {resultado['clasificaciones'][0]['confiabilidad']}%")
        else:
            print("❌ Error en respuesta")
            print(f"💬 Contenido: {respuesta.text}")
            
            # Intentar obtener JSON del error
            try:
                error_json = respuesta.json()
                print(f"🔍 Detalle del error: {error_json.get('detail', 'Sin detalle')}")
            except:
                pass
                
    except requests.exceptions.Timeout:
        print("⏰ Timeout - El servidor tardó demasiado")
    except requests.exceptions.ConnectionError:
        print("🔌 Error de conexión")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def verificar_logs_servidor():
    """
    📋 Instrucciones para verificar logs del servidor
    """
    print("=" * 60)
    print("📋 VERIFICAR LOGS DEL SERVIDOR")
    print("=" * 60)
    
    print("Para diagnosticar el error 500, revisa los logs del servidor:")
    print()
    print("1. 🖥️  En la terminal donde ejecutas 'python main.py'")
    print("2. 👀 Busca mensajes de error cuando envíes la imagen")
    print("3. 🔍 Los errores aparecerán como:")
    print("   - Traceback (most recent call last):")
    print("   - ERROR: Exception in ASGI application")
    print("   - Mensajes específicos de TensorFlow/PIL")
    print()
    print("4. 📋 Tipos de error comunes:")
    print("   - FileNotFoundError: Modelo no encontrado")
    print("   - ValueError: Error en forma de datos")
    print("   - PIL.UnidentifiedImageError: Imagen corrupta")
    print("   - TensorFlow errors: Problemas del modelo")
    print()

def probar_endpoint_info():
    """
    ℹ️ Verificar endpoints de información
    """
    print("=" * 60)
    print("ℹ️ VERIFICANDO ENDPOINTS DE INFO")
    print("=" * 60)
    
    endpoints = [
        ("/", "Inicio"),
        ("/salud/", "Salud del sistema"),
        ("/info/", "Información del modelo")
    ]
    
    for endpoint, descripcion in endpoints:
        try:
            respuesta = requests.get(f"{URL_BASE}{endpoint}")
            if respuesta.status_code == 200:
                print(f"✅ {descripcion:<20} - OK")
                if endpoint == "/salud/":
                    data = respuesta.json()
                    print(f"   Modelo cargado: {data.get('modelo_cargado', 'N/A')}")
            else:
                print(f"❌ {descripcion:<20} - Error {respuesta.status_code}")
        except Exception as e:
            print(f"🔌 {descripcion:<20} - Sin conexión")
    
    print()

def crear_imagen_test():
    """
    🎨 Crear una imagen de prueba simple para verificar el pipeline
    """
    print("=" * 60)
    print("🎨 CREANDO IMAGEN DE PRUEBA")
    print("=" * 60)
    
    try:
        from PIL import Image
        import numpy as np
        
        # Crear imagen RGB de prueba 224x224
        imagen_test = Image.new('RGB', (224, 224), color='red')
        ruta_test = Path("test_image.jpg")
        imagen_test.save(ruta_test, 'JPEG')
        
        print(f"✅ Imagen de prueba creada: {ruta_test}")
        print("📏 Tamaño: 224x224 (ideal para el modelo)")
        
        # Probar con esta imagen
        try:
            with open(ruta_test, 'rb') as archivo:
                archivos = {'archivo_imagen': archivo}
                datos = {'porcentaje_minimo': 5.0}
                
                print("⏳ Probando imagen sintética...")
                respuesta = requests.post(
                    f"{URL_BASE}/clasificar/",
                    files=archivos,
                    data=datos
                )
            
            if respuesta.status_code == 200:
                print("✅ ¡El pipeline funciona con imagen sintética!")
                resultado = respuesta.json()
                print(f"🏆 Resultado: {resultado['clasificaciones'][0]['nombre_raza']}")
            else:
                print(f"❌ Error con imagen sintética: {respuesta.status_code}")
                print(f"💬 {respuesta.text}")
                
        except Exception as e:
            print(f"❌ Error probando imagen sintética: {e}")
        
        # Limpiar archivo de prueba
        ruta_test.unlink()
        
    except ImportError:
        print("❌ PIL no disponible para crear imagen de prueba")
    except Exception as e:
        print(f"❌ Error creando imagen: {e}")

def main():
    """
    🎯 Ejecutar diagnóstico completo
    """
    print("🔧 DIAGNÓSTICO DEL BACKEND - CLASIFICADOR DE GANADO")
    print("Identificando la causa del error 500...")
    print()
    
    # 1. Verificar conectividad básica
    try:
        respuesta = requests.get(f"{URL_BASE}/")
        if respuesta.status_code == 200:
            print("✅ Servidor funcionando")
        else:
            print("❌ Servidor con problemas")
            return
    except:
        print("❌ No se puede conectar al servidor")
        print("   Asegúrate de ejecutar: python main.py")
        return
    
    # 2. Verificar archivos
    verificar_archivos_backend()
    
    # 3. Verificar endpoints de info
    probar_endpoint_info()
    
    # 4. Crear imagen de prueba
    crear_imagen_test()
    
    # 5. Probar con imagen real
    probar_imagen_simple()
    
    # 6. Instrucciones para logs
    verificar_logs_servidor()
    
    print()
    print("🎯 CONCLUSIÓN:")
    print("Si todo lo anterior está OK pero aún falla:")
    print("1. 👀 Revisa los logs del servidor en la terminal")
    print("2. 🔄 Reinicia el servidor (Ctrl+C y python main.py)")
    print("3. 🐍 Verifica la versión de Python y librerías")
    print("4. 💾 Asegúrate de que hay suficiente memoria RAM")

if __name__ == "__main__":
    main()