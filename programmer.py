# programmer.py

from ollama import Client
import ast
import re
import logging
import random
import pickle
import os

client = Client(host='http://localhost:11434')

class Programmer:
    def __init__(self, prompt_master, epsilon=0.1):
        self.q_table = {}
        self.prompt = (
            "Você é um programador experiente em ciência de dados. Escreva apenas o código, sem texto adicional, "
            "rótulos de linguagem, cabeçalhos ou explicações. Você pode adicionar comentários para legibilidade. "
            "O código deve incluir todas as importações necessárias e ser executável como está. Se estiver criando uma função, forneça um exemplo de uso no final."
        )
        self.current_prompt = ""
        self.hints = ""
        self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
        self.prompt_history = []
        self.code_history = []
        self.reward_history = []
        self.max_attempts = 10
        self.epsilon = epsilon  # Probabilidade de explorar ações
        self.q_table = {}  # Tabela Q para armazenar valores de ações
        self.prompt_master = prompt_master  # Instância de PromptMaster
        self.programmer_reward_history = []
        self.programmer_weights_history = []

    def _set_current_prompt(self, question, hint=None):
        self.current_prompt = self.prompt + "\n\n"
        self.current_prompt += f"Considere os seguintes pesos ao escrever o código:\n{self.weights}\n\n"
        if hint:
            self.current_prompt += f"Dica: {hint}\n\n"
        self.current_prompt += f"Agora, seguindo todas as regras anteriores, escreva um código para responder à seguinte questão:\n{question}"
        self.prompt_history.append(self.current_prompt)

    def act(self, question, training=True):
        state = self.get_state()
        
        # Seleção de ação baseada na política epsilon-greedy
        if training and random.random() < self.epsilon:
            hint, hint_strength, weights = self.prompt_master.create_hint('CODE', '', '', self.get_last_score(), self.weights)
            action = f"Dica: {hint}"
            logging.info(f"Programador explorando com ação: {action}")
        else:
            action = self.exploit(state)
            logging.info(f"Programador explorando com ação: {action}")
            hint, hint_strength, weights = self.prompt_master.extract_hint(action)
        
        # Atualizar pesos se necessário
        if weights:
            self.weights = weights
        
        # Configurar o prompt com a dica
        self._set_current_prompt(question, hint)
        
        # Gerar o código com o prompt completo
        response = self.generate_code(self.current_prompt)
        code = response['message']['content']
        self.code_history.append(code)
        return code

    def generate_code(self, prompt):
        attempts = 0
        response = {'done_reason': None}
        full_code = ""
        while attempts < self.max_attempts:
            response = client.chat(model='llama3.1', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            content = response['message']['content']
            if '...' in content:
                full_code += content.replace('...', '')
                prompt += "\nPor favor, complete o código acima."
            else:
                full_code += content
                break
            attempts += 1
        return {'message': {'content': full_code}}


    def get_state(self):
        # Define o estado como a pontuação atual do código (0-1)
        if not self.code_history:
            return 0
        last_score = self.reward_history[-1] if self.reward_history else 0
        return last_score

    def explore(self):
        # Seleciona uma ação (prompt) aleatória da tabela Q
        if not self.q_table:
            return self.prompt_master.get_random_action('CODE')
        state = self.get_state()
        if state in self.q_table and self.q_table[state]:
            return max(self.q_table[state], key=self.q_table[state].get)
        else:
            return self.prompt_master.get_random_action('CODE')

    def exploit(self, state):
        # Seleciona a melhor ação conhecida para o estado atual
        if state in self.q_table and self.q_table[state]:
            action = max(self.q_table[state], key=self.q_table[state].get)
            return action
        else:
            return self.explore()

    def update_policy(self, state, action, reward):
        # Atualiza a tabela Q usando a fórmula Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        alpha = 0.1  # Taxa de aprendizado
        gamma = 0.9  # Fator de desconto

        if state not in self.q_table:
            self.q_table[state] = {}

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0

        max_future_q = max(self.q_table.get(state, {}).values()) if self.q_table.get(state, {}) else 0

        current_q = self.q_table[state][action]
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

        logging.info(f"Atualizado Q({state}, {action}) = {self.q_table[state][action]} com recompensa {reward}")

    def get_last_score(self):
        return self.reward_history[-1] if self.reward_history else 0

    def update(self, new_hint, hint_weight, weights):
        if new_hint:
            self.hints += f"\n- {new_hint} (Peso: {hint_weight})"
        if weights:
            self.weights = weights
        self._store_hints()

    def _store_hints(self):
        # Armazena as dicas e pesos
        text = str(self.weights) + "\n" + self.hints
        with open("data/programmer_hints.txt", "w") as file:
            file.write(text)
        logging.info("Dicas e pesos do programador armazenados.")

    def load_hints(self):
        try:
            with open("data/programmer_hints.txt", "r") as file:
                lines = file.readlines()
            if lines:
                self.weights = ast.literal_eval(lines[0].strip())
                self.hints = "".join(lines[1:]).strip()
        except (SyntaxError, ValueError, FileNotFoundError) as e:
            logging.error(f"Erro ao carregar dicas do programador: {e}")
            self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
            self.hints = ""

    def reset(self):
        # Resetar histórico após um ciclo de treinamento
        self.prompt_history = []
        self.code_history = []
        self.reward_history = []

    def save(self, filepath):
        """Salva o estado atual do agente Programmer em um arquivo."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Programador salvo em {filepath}")

    @staticmethod
    def load(filepath):
        """Carrega o estado do agente Programmer a partir de um arquivo."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                programmer = pickle.load(f)
            logging.info(f"Programador carregado de {filepath}")
            return programmer
        else:
            logging.warning(f"Arquivo {filepath} não encontrado. Inicializando um novo Programador.")
            return None