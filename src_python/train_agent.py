import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from bb_env import BBEnv

# Silenciar logs innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# 1. DEFINICIÓN DEL MLP (RED NEURONAL)
# ==========================================
class DQN_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_MLP, self).__init__()
        # Arquitectura feed-forward clásica
        self.red = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Una neurona por cada acción (Valor Q)
        )

    def forward(self, x):
        return self.red(x)

# ==========================================
# 2. LA MEMORIA (REPLAY BUFFER)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacidad_maxima):
        self.buffer = deque(maxlen=capacidad_maxima)

    def guardar_experiencia(self, estado, accion, recompensa, siguiente_estado, done):
        # Almacenamos la transición en la memoria
        self.buffer.append((estado, accion, recompensa, siguiente_estado, done))

    def muestrear_batch(self, batch_size):
        # Seleccionamos un grupo de experiencias al azar
        lote = random.sample(self.buffer, batch_size)
        estados, acciones, recompensas, siguientes_estados, dones = zip(*lote)
        return (
            np.array(estados),
            np.array(acciones),
            np.array(recompensas, dtype=np.float32),
            np.array(siguientes_estados),
            np.array(dones, dtype=np.float32)
        )

    def tamano(self):
        return len(self.buffer)

# ==========================================
# 3. EL AGENTE DQN
# ==========================================
class AgenteDQN:
    def __init__(self, input_dim, num_acciones):
        self.num_acciones = num_acciones
        
        # Red Principal (la que entrena)
        self.modelo = DQN_MLP(input_dim, num_acciones)
        # Red Target (Copia estable para calcular el target de la Ecuación de Bellman)
        self.modelo_target = DQN_MLP(input_dim, num_acciones)
        self.modelo_target.load_state_dict(self.modelo.state_dict())
        self.modelo_target.eval() # La red target no acumula gradientes
        
        # Optimizador (Aplica los ajustes físicos a los pesos)
        self.optimizador = optim.Adam(self.modelo.parameters(), lr=0.001)
        # Función de pérdida (Error Cuadrático Medio)
        self.loss_fn = nn.MSELoss()
        
        self.memoria = ReplayBuffer(capacidad_maxima=10000)
        
        # Parámetros del algoritmo
        self.batch_size = 64
        self.gamma = 0.99       # Factor de descuento futuro
        self.epsilon = 1.0      # Empieza 100% explorando al azar
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def elegir_accion(self, estado):
        # Política Epsilon-Greedy: Balance entre explorar y explotar
        if random.random() < self.epsilon:
            # Acción al azar
            return random.randint(0, self.num_acciones - 1)
        else:
            # Acción inteligente basada en el MLP
            estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
            with torch.no_grad():
                valores_q = self.modelo(estado_tensor)
            return valores_q.argmax().item()

    def aprender_de_memoria(self):
        # Solo entrena si hay suficientes datos en la memoria
        if self.memoria.tamano() < self.batch_size:
            return None

        # 1. Extraer un Mini-Batch aleatorio
        estados, acciones, recompensas, sig_estados, dones = self.memoria.muestrear_batch(self.batch_size)

        # Convertir todo a tensores de PyTorch
        estados_t = torch.FloatTensor(estados)
        acciones_t = torch.LongTensor(acciones).unsqueeze(1)
        recompensas_t = torch.FloatTensor(recompensas).unsqueeze(1)
        sig_estados_t = torch.FloatTensor(sig_estados)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # 2. EL MLP PREDICE: Valores Q para las acciones que realmente tomó
        q_predichos = self.modelo(estados_t).gather(1, acciones_t)

        # 3. ECUACIÓN DE BELLMAN: Calcular el Target real
        with torch.no_grad():
            q_sig_max = self.modelo_target(sig_estados_t).max(1)[0].unsqueeze(1)
            # Si el episodio terminó (done=1), el valor futuro es 0
            targets = recompensas_t + (self.gamma * q_sig_max * (1 - dones_t))

        # 4. CALCULAR ERROR (Loss)
        loss = self.loss_fn(q_predichos, targets)

        # 5. ¡RETROPROPAGACIÓN! Entregar el error al MLP
        self.optimizador.zero_grad()
        loss.backward()
        self.optimizador.step()
        
        return loss.item()

    def actualizar_red_target(self):
        # Copia los pesos aprendidos a la red de referencia
        self.modelo_target.load_state_dict(self.modelo.state_dict())

    def reducir_exploracion(self):
        # Disminuye lentamente el epsilon para que la red tome más el control
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 4. BUCLE PRINCIPAL DE ENTRENAMIENTO
# ==========================================
def main():
    print("==================================================")
    print(" INICIANDO ENTRENAMIENTO DE HIPER-HEURÍSTICA RL")
    print(" Arquitectura: DQN con Replay Buffer (PyTorch)")
    print("==================================================")
    print("Abre otra terminal y ejecuta Ibex en bucle continuo:")
    print("while true; do bin/ibex_sockets ../benchs/optim/benchs_victor_fixxed/disc2; done")
    print("==================================================")

    # Iniciar Entorno (5 características, 4 acciones)
    env = BBEnv(host='127.0.0.1', port=8888)
    agente = AgenteDQN(input_dim=5, num_acciones=4)

    episodios_completados = 0
    pasos_totales = 0

    try:
        while True: # Bucle infinito esperando problemas de Ibex
            
            estado, _ = env.reset()
            episodio_recompensa = 0
            loss_promedio = []
            
            # Bucle dentro de un solo problema de optimización
            while True:
                # 1. Agente decide acción
                accion = agente.elegir_accion(estado)
                print(f"Acción elegida por la red: {accion}") # Mira si cambia con el tiempo
                
                # 2. Entorno ejecuta y responde
                siguiente_estado, recompensa, done, _, _ = env.step(accion)
                
                # 3. Guardar en memoria (Replay Buffer)
                agente.memoria.guardar_experiencia(estado, accion, recompensa, siguiente_estado, done)
                
                # 4. ¡Entregar el batch de errores al MLP!
                loss = agente.aprender_de_memoria()
                if loss is not None:
                    loss_promedio.append(loss)
                
                estado = siguiente_estado
                episodio_recompensa += recompensa
                pasos_totales += 1
                
                # Actualizar la Red Target periódicamente (ej. cada 100 pasos)
                if pasos_totales % 100 == 0:
                    agente.actualizar_red_target()


                if done:
                    break # Problema resuelto, salir al bucle principal

            # Fin del Episodio (Ibex resolvió el problema)
            episodios_completados += 1
            agente.reducir_exploracion()
            
            avg_loss = np.mean(loss_promedio) if len(loss_promedio) > 0 else 0.0
            print(f"Episodio {episodios_completados} | Recompensa Total: {episodio_recompensa:.2f} | Epsilon: {agente.epsilon:.2f} | Loss: {avg_loss:.4f}")

            # Guardar el modelo cada 50 episodios
            if episodios_completados % 50 == 0:
                torch.save(agente.modelo.state_dict(), f"modelo_dqn_ep{episodios_completados}.pth")
                print(f"[GUARDADO] Modelo en episodio {episodios_completados}")

    except KeyboardInterrupt:
        print("\n[AVISO] Entrenamiento interrumpido. Guardando progreso final...")
        torch.save(agente.modelo.state_dict(), "modelo_dqn_final.pth")
    finally:
        env.close()

if __name__ == "__main__":
    main()