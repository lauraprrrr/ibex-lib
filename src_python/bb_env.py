import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json

class BBEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=8888, recv_timeout=60.0):
        super(BBEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        self.host = host
        self.port = port
        # Timeout SOLO para el socket por-episodio (self.conn), una vez conectado.
        # Acota cuánto esperamos a Ibex a mitad de un episodio (proceso colgado,
        # etc). Deliberadamente NO se aplica al socket de escucha (ver _init_socket).
        self.recv_timeout = recv_timeout
        self.server_socket = None
        self.conn = None
        self.addr = None

    def _init_socket(self):
        if self.server_socket is None:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            # Sin timeout aquí a propósito. accept() espera a que (re)lances el
            # binario de Ibex en otra terminal -- eso puede tardar segundos o
            # minutos según cómo lo operes, y no es un "cuelgue" a detectar.
            # Poner un timeout aquí == el entrenamiento se cae en el primer
            # reset() si no arrancas Ibex a tiempo.
            print(f"[BBEnv] Escuchando en {self.host}:{self.port}...")

    def _wait_for_connection(self):
        self._init_socket()
        if self.conn is None:
            print("[BBEnv] Esperando conexión de Ibex (lanza/relanza el binario ahora)...")
            self.conn, self.addr = self.server_socket.accept()
            # El timeout por-mensaje SÍ aplica desde aquí en adelante: acota
            # cuánto esperamos a Ibex a mitad de un episodio ya iniciado.
            self.conn.settimeout(self.recv_timeout)
            print(f"[BBEnv] Ibex conectado desde {self.addr}")

    def _receive_state_from_ibex(self):
        buffer = ""
        while True:
            try:
                data = self.conn.recv(4096).decode('utf-8')
                if not data:
                    # [CLAVE] Forzar olvido de conexión si C++ cortó
                    self.conn.close()
                    self.conn = None
                    return None, 0.0, True 
                
                buffer += data
                if '\n' in buffer: 
                    line, buffer = buffer.split('\n', 1)
                    state_dict = json.loads(line)
                    
                    obs = np.array(state_dict["features"], dtype=np.float32)
                    obs = np.clip(obs, -100.0, 100.0)
                    
                    reward = float(state_dict["reward"])
                    reward = np.clip(reward, -50.0, 50.0)
                    done = bool(state_dict["done"])
                    
                    return obs, reward, done
                    
            except socket.timeout:
                return None, 0.0, True
            except json.JSONDecodeError as e:
                print(f"[DEBUG JSON ERROR]: {e} | Trama recibida: {line}")
                return None, 0.0, True

    def reset(self, seed=None, options=None):
            """
            Inicia un nuevo episodio. Se bloquea esperando a que una nueva
            instancia del solver Ibex se conecte y envíe su estado inicial (raíz).
            """
            super().reset(seed=seed)
            
            # 1. Asegurar que estamos escuchando/conectados
            self._wait_for_connection()
            
            # 2. Recibir el primer estado (Que ahora C++ manda desde Optimizer::start)
            obs, reward, done = self._receive_state_from_ibex()
            
            # Fallback de seguridad por si el socket falló en el microsegundo de conexión
            if obs is None:
                print("[BBEnv] Aviso: obs None en reset, devolviendo ceros.")
                obs = np.zeros(5, dtype=np.float32)
                
            # Gym espera devolver (observación, info_dict)
            return obs, {}

    def step(self, action):
            # Verificar si la conexión se perdió antes de enviar
            if self.conn is None:
                print("[BBEnv] Aviso: Conexión perdida antes de enviar acción.")
                return np.zeros(5, dtype=np.float32), 0.0, True, False, {}

            msg = json.dumps({"decision": int(action)}) + "\n"
            try:
                self.conn.send(msg.encode('utf-8'))
            except (BrokenPipeError, AttributeError):
                print("[BBEnv] Pipe roto al enviar acción.")
                if self.conn:
                    self.conn.close()
                self.conn = None
                return np.zeros(5, dtype=np.float32), 0.0, True, False, {}

            # Doble verificación antes de recibir para evitar el AttributeError
            if self.conn is None:
                return np.zeros(5, dtype=np.float32), 0.0, True, False, {}

            obs, reward, done = self._receive_state_from_ibex()
            
            if done or obs is None:
                if self.conn:
                    try:
                        msg_ok = json.dumps({"decision": 0}) + "\n"
                        self.conn.send(msg_ok.encode('utf-8'))
                        self.conn.close()
                    except:
                        pass
                self.conn = None 
                return np.zeros(5, dtype=np.float32), reward, True, False, {}

            return obs, reward, done, False, {}

    def close(self):
        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()