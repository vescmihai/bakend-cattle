from ultralytics import YOLO
import os

def main():
    # Cargar modelo YOLO
    print("🔄 Cargando modelo YOLO...")
    model = YOLO('yolov8n.pt')
    print("✅ Modelo cargado")
    
    while True:
        # Pedir ruta de imagen
        print("-" * 40)
        ruta_imagen = input("📁 Ruta de la imagen (o 'salir'): ").strip()
        
        if ruta_imagen.lower() in ['salir', 'exit', 'quit', 's']:
            print("👋 ¡Hasta luego!")
            break
        
        if not ruta_imagen:
            print("⚠️ Ingresa una ruta válida")
            continue
            
        if not os.path.exists(ruta_imagen):
            print("❌ Archivo no encontrado")
            continue
        
        # Detectar objetos
        print("🔍 Analizando imagen...")
        results = model(ruta_imagen)
        
        # Lista de animales que detecta YOLO
        animales = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                   'elephant', 'bear', 'zebra', 'giraffe']
        
        es_animal = False
        es_vaca = False
        animales_detectados = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Verificar si es animal con confianza > 60%
                    if class_name in animales and confidence > 0.6:
                        es_animal = True
                        animales_detectados.append((class_name, confidence))
                        
                        # Verificar si específicamente es vaca
                        if class_name == 'cow':
                            es_vaca = True
        
        # Mostrar resultados
        print("\n🎯 RESULTADOS:")
        print("=" * 30)
        
        if not es_animal:
            print("❌ NO es un animal")
        elif es_vaca:
            print("✅ ES un animal")
            print("🐄 ES una VACA")
        else:
            print("✅ ES un animal")
            print("❌ NO es una vaca")
        
        # Mostrar detecciones con porcentajes
        if animales_detectados:
            print(f"\nDetecciones:")
            for animal, confianza in animales_detectados:
                print(f"  • {animal}: {confianza:.1%}")
        
        # Preguntar si continuar
        continuar = input(f"\n¿Analizar otra imagen? (s/n): ").lower()
        if continuar not in ['s', 'si', 'sí', 'y', 'yes']:
            print("👋 ¡Hasta luego!")
            break

if __name__ == "__main__":
    main()