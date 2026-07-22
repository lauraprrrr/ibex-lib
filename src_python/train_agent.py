import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import deque
from bb_env import BBEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RewardNormalizer:
    def __init__(self, clip_val=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.clip_val = clip_val

    def normalize(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2
        std = math.sqrt(self.var / self.count) + 1e-8
        normalized_reward = (reward - self.mean) / std
        return np.clip(normalized_reward, -self.clip_val, self.clip_val)
    
class DQN_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_MLP, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.red(x)

class ReplayBuffer:
    def __init__(self, capacidad_maxima):
        self.buffer = deque(maxlen=capacidad_maxima)

    def guardar_experiencia(self, estado, accion, recompensa, siguiente_estado, done):
        self.buffer.append((estado, accion, recompensa, siguiente_estado, done))

    def muestrear_batch(self, batch_size):
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

class AgenteDQN:
    def __init__(self, input_dim, num_acciones):
        self.num_acciones = num_acciones
        self.modelo = DQN_MLP(input_dim, num_acciones)
        self.modelo_target = DQN_MLP(input_dim, num_acciones)
        self.modelo_target.load_state_dict(self.modelo.state_dict())
        self.modelo_target.eval()
        
        self.optimizador = optim.Adam(self.modelo.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memoria = ReplayBuffer(capacidad_maxima=10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def elegir_accion(self, estado):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_acciones - 1)
        else:
            estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
            with torch.no_grad():
                valores_q = self.modelo(estado_tensor)
            return valores_q.argmax().item()

    def aprender_de_memoria(self):
        if self.memoria.tamano() < self.batch_size:
            return None

        estados, acciones, recompensas, sig_estados, dones = self.memoria.muestrear_batch(self.batch_size)
        estados_t = torch.FloatTensor(estados)
        acciones_t = torch.LongTensor(acciones).unsqueeze(1)
        recompensas_t = torch.FloatTensor(recompensas).unsqueeze(1)
        sig_estados_t = torch.FloatTensor(sig_estados)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        q_predichos = self.modelo(estados_t).gather(1, acciones_t)

        with torch.no_grad():
            q_sig_max = self.modelo_target(sig_estados_t).max(1)[0].unsqueeze(1)
            targets = recompensas_t + (self.gamma * q_sig_max * (1 - dones_t))

        loss = self.loss_fn(q_predichos, targets)
        self.optimizador.zero_grad()
        loss.backward()
        self.optimizador.step()
        
        return loss.item()

    def actualizar_red_target(self):
        self.modelo_target.load_state_dict(self.modelo.state_dict())

    def reducir_exploracion(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    print("==================================================")
    print(" INICIANDO ENTRENAMIENTO DE HIPER-HEURÍSTICA RL")
    print("==================================================")

    env = BBEnv(host='127.0.0.1', port=8888)
    agente = AgenteDQN(input_dim=5, num_acciones=4)
    reward_norm = RewardNormalizer(clip_val=5.0)

    episodios_completados = 0
    pasos_totales = 0

    try:
        while True: 
            
            estado, _ = env.reset()
            episodio_recompensa = 0
            loss_promedio = []
            done = False
            
            while not done:
                # 1. Agente decide
                accion = agente.elegir_accion(estado)
                
                # 2. Enviar a C++ y esperar resultados
                siguiente_estado, recompensa_cruda, done, _, _ = env.step(accion)
                # print(f"Estado recibido de C++: {siguiente_estado}")
                
                # 3. Normalizar Recompensa
                if not done:
                    recompensa = reward_norm.normalize(recompensa_cruda)
                else:
                    recompensa = recompensa_cruda 
                # 4. Memoria y Aprendizaje
                agente.memoria.guardar_experiencia(estado, accion, recompensa, siguiente_estado, done)
                loss = agente.aprender_de_memoria()
                if loss is not None:
                    loss_promedio.append(loss)
                
                estado = siguiente_estado
                episodio_recompensa += recompensa
                pasos_totales += 1
                
                if pasos_totales % 100 == 0:
                    agente.actualizar_red_target()

            # --- Fin del Episodio ---
            episodios_completados += 1
            agente.reducir_exploracion()
            
            avg_loss = np.mean(loss_promedio) if len(loss_promedio) > 0 else 0.0
            print(f"Ep. {episodios_completados} | Reward: {episodio_recompensa:.2f} | Eps: {agente.epsilon:.2f} | Loss: {avg_loss:.4f}")

            if episodios_completados % 50 == 0:
                torch.save(agente.modelo.state_dict(), f"modelo_dqn_ep{episodios_completados}.pth")

    except KeyboardInterrupt:
        print("\n[AVISO] Guardando progreso final...")
        torch.save(agente.modelo.state_dict(), "modelo_dqn_final.pth")
    finally:
        env.close()

if __name__ == "__main__":
    main()