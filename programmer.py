# programmer.py

from ollama import Client
import ast
import re
import logging
import random

client = Client(host='http://localhost:11434')

class Programmer:
    def __init__(self, epsilon=0.1):
        self.prompt = ("Você é um programador experiente em ciência de dados. Escreva apenas o código, sem texto adicional, "
                      "rótulos de linguagem, cabeçalhos ou explicações. Você pode adicionar comentários para legibilidade. "
                      "O código deve incluir todas as importações necessárias e ser executável como está. Se estiver criando uma função, forneça um exemplo de uso no final.")
        self.current_prompt = ""
        self.hints = ""
        self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
        self.prompt_history = []
        self.code_history = []
        self.reward_history = []
        self.max_attempts = 10
        self.epsilon = epsilon  # Probabilidade de explorar ações
        self.q_table = {}  # Tabela Q para armazenar valores de ações

    def _set_current_prompt(self, question):
        self.current_prompt = self.prompt + "\nConsidere os seguintes pesos ao escrever o código:\n"
        self.current_prompt += str(self.weights)
        self.current_prompt += "\nConsidere as seguintes dicas (hints), cada uma com um peso de 1 a 100, indicando sua importância. As dicas são:"
        self.current_prompt += self.hints
        self.current_prompt += "\nAgora, seguindo todas as regras anteriores, escreva um código para responder à seguinte questão:\n"
        self.current_prompt += question
        self.prompt_history.append(self.current_prompt)

    def act(self, question, training=True):
        self._set_current_prompt(question)
        state = self.get_state()
        
        # Seleção de ação baseada na política epsilon-greedy
        if training and random.random() < self.epsilon:
            action = self.explore()
            logging.info(f"Programador explorando com ação: {action}")
        else:
            action = self.exploit(state)
            logging.info(f"Programador explorando com ação: {action}")
        
        response = self.generate_code(action)
        code = response['message']['content']
        self.code_history.append(code)
        return code

    def generate_code(self, prompt):
        attempts = 0
        response = {'done_reason': None}
        while response['done_reason'] != 'stop' and attempts < self.max_attempts:
            response = client.chat(model='llama3.1', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            attempts += 1
        return response

    def get_state(self):
        # Define o estado como a pontuação atual do código (0-1)
        if not self.code_history:
            return 0
        last_score = self.reward_history[-1] if self.reward_history else 0
        return last_score

    def explore(self):
        # Seleciona uma ação (prompt) aleatória
        # Neste contexto, a ação é o prompt master para gerar dicas
        hint, hint_strength, weights = self.generate_hint()
        action = f"Dica: {hint}"
        return action

    def exploit(self, state):
        # Seleciona a melhor ação conhecida para o estado atual
        if state in self.q_table and self.q_table[state]:
            action = max(self.q_table[state], key=self.q_table[state].get)
            return action
        else:
            return self.explore()

    def generate_hint(self):
        # Implementação para gerar uma dica (hint) usando PromptMaster
        # Placeholder: substitua com chamada real
        hint = "Melhore a legibilidade do código adicionando comentários explicativos."
        hint_strength = 80
        weights = self.weights
        return hint, hint_strength, weights

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
