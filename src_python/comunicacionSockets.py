# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL: SILENCIAR LOGS Y ADVERTENCIAS
# --------------------------------------------------------------------------
import os
import warnings
import sys

# Silenciar logs de bajo nivel (C++) de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Opcional: Forzar el uso de la CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Suprimir advertencias de Python
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# 2. IMPORTACIÓN DE LIBRERÍAS
# --------------------------------------------------------------------------
import json
import numpy as np
import joblib  # Necesario para el Scaler
import socket
import tensorflow as tf
import threading
from keras.models import load_model
from concurrent.futures import ThreadPoolExecutor

MAX_WORKERS = min(32, (os.cpu_count() or 1) + 1)

# --------------------------------------------------------------------------
# 3. CARGA DEL MODELO Y SCALER (Una sola vez)
# --------------------------------------------------------------------------

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_PATH = os.path.join(script_dir, '/home/laura/Documents/uni/investigacion/codigo/bb_project/ibex-lib/src_python/Arquitecturas/mlp_3_tanh_temperature.keras')
    SCALER_PATH = os.path.join(script_dir, '/home/laura/Documents/uni/investigacion/codigo/bb_project/ibex-lib/src_python/Arquitecturas/scaler_3_tanh_temperature.pkl')

    # --- PASO A: Definir la función personalizada EXACTAMENTE igual al entrenamiento ---
    def softmax_temperature(logits):
        return tf.nn.softmax(logits / 0.8)

    print(f"[INFO] Cargando modelo desde: {MODEL_PATH}")
    
    # --- PASO B: Cargar el modelo pasando 'custom_objects' ---
    model = load_model(MODEL_PATH, custom_objects={'softmax_temperature': softmax_temperature})
    
    print(f"[INFO] Cargando scaler desde: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    print("[OK] Modelo y Scaler cargados exitosamente")
    
except Exception as e:
    print(f'[FATAL] Error al cargar archivos: {e}', file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------------------------------
# 4. FUNCIÓN PARA PROCESAR PREDICCIONES
# --------------------------------------------------------------------------
def process_prediction(input_json):

    try:
        # 1. Decodificar JSON
        data = json.loads(input_json)
        features_vector = data['features']
        
        # 2. Convertir a Numpy Array
        features_raw = np.array(features_vector).reshape(1, -1)

        # Validación de tamaño (Tu modelo espera 7 inputs)
        if len(features_vector) != 7:
            raise ValueError(f"Se esperaba vector de 7, llegaron {len(features_vector)}")
        
        # --- CAMBIO 2: APLICAR EL SCALER (Normalización) ---
        features_norm = scaler.transform(features_raw)
        
        # 3. Realizar la predicción
        # La salida son probabilidades directas (Softmax)
        probabilities = model.predict(features_norm, verbose=0)[0]

        # --- CAMBIO 3: SELECCIÓN ESTOCÁSTICA (SIN TEMPERATURA) ---
        # Usamos las probabilidades tal cual salen de la red para tirar los dados.
        # np.random.choice elige un índice (0-5) respetando los porcentajes de 'p'
        
        opciones = np.arange(len(probabilities))
        decision_index = int(np.random.choice(opciones, p=probabilities))

        # Debug en consola (Opcional, para que veas qué pasa)
        # print(f"Probs: {np.round(probabilities, 2)} -> Ganador: {decision_index}")

        # 4. Preparar la respuesta JSON
        response = {
            'decision': decision_index,
            'id': data.get('id', -1),
            # Opcional: devolver probabilidades por si quieres debuggear en C++
            'probabilities': probabilities.tolist() 
        }
        
        return json.dumps(response)    
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        error_id = -1
        try:
            data = json.loads(input_json)
            error_id = data.get('id', -1)
        except: pass
            
        response = { 'error': f'Error de datos: {e}', 'id': error_id }
        return json.dumps(response)
        
    except Exception as e:
        error_id = -1
        try:
            data = json.loads(input_json)
            error_id = data.get('id', -1)
        except: pass
            
        response = { 'error': f'Error inesperado: {str(e)}', 'id': error_id }
        return json.dumps(response)

# --------------------------------------------------------------------------
# 5. MANEJADOR DE CLIENTE 
# --------------------------------------------------------------------------
def handle_client(client_socket, address):
    try:
        while True:
            # Recibir datos (Buffer de 4096 es más seguro para JSON)
            data = client_socket.recv(4096)
            if not data:
                break
                
            input_json = data.decode('utf-8')
            response = process_prediction(input_json)
            client_socket.send(response.encode('utf-8'))
            
    except Exception as e:
        print(f"Error cliente {address}: {e}")
        try:
            client_socket.send(json.dumps({'error': str(e)}).encode('utf-8'))
        except: pass
    finally:
        client_socket.close()

# --------------------------------------------------------------------------
# 6. SERVIDOR PRINCIPAL
# --------------------------------------------------------------------------
def start_server(host='localhost', port=8888):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            server_socket.bind((host, port))
            server_socket.listen(10)
            print(f"[LISTO] Servidor JSON escuchando en {host}:{port}")
            
            while True:
                client_socket, address = server_socket.accept()
                executor.submit(handle_client, client_socket, address)
                
        except KeyboardInterrupt:
            print("\nCerrando servidor...")
        except Exception as e:
            print(f"Error servidor: {e}")
        finally:
            server_socket.close()

# ... (Todo el código de imports, clases y funciones predecir/manejar_cliente igual al anterior) ...

# --------------------------------------------------------------------------
# 7. MAIN (VERSIÓN ROBUSTA/REFERENCIA)
# --------------------------------------------------------------------------
def main():
    """
    Función principal del servidor
    """
    # Configuración por defecto
    HOST = 'localhost'
    PORT = 8888
    
    # 1. Verificar Puerto (Argumento 1)
    if len(sys.argv) > 1:
        try:
            PORT = int(sys.argv[1])
        except ValueError:
            print(f"[AVISO] Puerto '{sys.argv[1]}' inválido. Usando defecto: {PORT}")
    
    # 2. Verificar Host (Argumento 2) - Útil si corres Ibex en otra máquina
    if len(sys.argv) > 2:
        HOST = sys.argv[2]
    
    print(f"--- Configuración: {HOST}:{PORT} ---")
    start_server(HOST, PORT)

if __name__ == '__main__':
    main()