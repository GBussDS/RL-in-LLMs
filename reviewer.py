# reviewer.py

from ollama import Client
import ast
import re
import pickle
import os
import logging
import random

client = Client(host='http://localhost:11434')

class Reviewer:
    def __init__(self, prompt_master, epsilon=0.1):
        self.q_table = {}
        self.review_prompt = (
            "Você é um revisor de código especialista em ciência de dados. Realize uma revisão abrangente do código fornecido, "
            "avaliando clareza, legibilidade, eficiência e otimização. Identifique erros, sugira melhorias e adicione comentários quando necessário. "
            "No FINAL da sua revisão, forneça uma pontuação total de 1 a 100, bem como pontuações individuais para clareza, legibilidade, eficiência e otimização, no seguinte formato:\n"
            "{'Total': score, 'clarity': score, 'readability': score, 'efficiency': score, 'optimization': score}"
        )
        self.report_prompt = (
            "Você é um analista de dados especialista encarregado de criar um relatório analítico abrangente com base no código fornecido. "
            "O relatório deve incluir etapas de limpeza de dados, transformações aplicadas, insights derivados e quaisquer visualizações geradas. "
            "Assegure-se de que o relatório seja claro, conciso e bem estruturado, adequado para que as partes interessadas compreendam o fluxo de trabalho dos dados e os resultados. "
            "No FINAL do seu relatório, forneça uma pontuação geral de qualidade de 1 a 100 no seguinte formato:\n"
            "{'Report Quality': score}"
        )
        self.current_prompt = ""
        self.hints = ""
        self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
        self.prompt_history = []
        self.review_history = []
        self.report_history = []
        self.reward_history = []
        self.max_attempts = 10
        self.epsilon = epsilon  # Probabilidade de explorar ações
        self.q_table = {}  # Tabela Q para armazenar valores de ações
        self.prompt_master = prompt_master  # Instância de PromptMaster

        # Adicionando os atributos faltantes
        self.reviewer_reward_history = []
        self.reviewer_weights_history = []

    def _set_current_prompt(self, stage, code, review):
        self.current_prompt = stage + self.review_prompt
        self.current_prompt += "\n\nO código é:\n\n" + code
        self.current_prompt += "\n\nA revisão foi:\n\n" + review

        self.current_prompt += "\n\nPara referência, estes são os pesos que o " + stage + " usou e a pontuação subsequente que alcançou:" 

        if stage == 'CODE':
            for reward, weight in zip(self.programmer_reward_history, self.programmer_weights_history):
                self.current_prompt += "\n" + "Pontuação: "+ str(reward) + " Pesos: " + str(weight)
        elif stage == 'REVIEW':
            for reward, weight in zip(self.reviewer_reward_history, self.reviewer_weights_history):
                self.current_prompt += "\n" + "Pontuação: "+ str(reward) + " Pesos: " + str(weight)

    def act(self, code, training=True):
        stage = 'REVIEW'
        self._set_current_prompt(stage, code, '')
        state = self.get_state(stage)
        
        # Seleção de ação baseada na política epsilon-greedy
        if training and random.random() < self.epsilon:
            hint, hint_strength, weights = self.prompt_master.create_hint('REVIEW', code, '', self.get_last_score(), self.weights)
            action = f"Dica: {hint}"
            logging.info(f"Revisor explorando com ação: {action}")
        else:
            action = self.exploit(stage, state)
            logging.info(f"Revisor explorando com ação: {action}")
            hint, hint_strength, weights = self.prompt_master.extract_hint(action)
        
        # Atualizar pesos se necessário
        if weights:
            self.weights = weights
        
        # Configurar o prompt com a dica
        self._set_current_prompt(stage, code, hint)
        
        # Gerar a revisão com o prompt completo
        response = self.generate_review(self.current_prompt)
        review, score = self.extract_scores(response['message']['content'])
        self.review_history.append(review)
        self.reward_history.append(score)
        return action, review, score

    def extract_scores(self, text):
        score_match = re.search(r"\{.*\}", text)
        if score_match:
            score_text = score_match.group()
            try:
                score = ast.literal_eval(score_text)
            except Exception as e:
                logging.error(f"Erro ao interpretar a pontuação: {e}")
                score = {'Total': 0, 'clarity': 0, 'readability': 0, 'efficiency': 0, 'optimization': 0}
            review = text[:score_match.start()].strip()
            return review, score
        else:
            logging.error("Formato de pontuação inválido recebido.")
            return text, {'Total': 0, 'clarity': 0, 'readability': 0, 'efficiency': 0, 'optimization': 0}


    def generate_review(self, prompt):
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

    def act_report(self, code, training=True):
        # Implementação similar para gerar relatórios
        # Placeholder: substituir com lógica real
        report = "Relatório gerado com sucesso."
        report_score = 85
        return report, report_score

    def generate_report(self, code):
        stage = 'REVIEW'
        self.current_prompt = self.report_prompt + "\n\n" + code
        response = self.generate_report_prompt(self.current_prompt)
        report, quality_score = self.extract_report_score(response['message']['content'])
        self.report_history.append(report)
        return report, quality_score

    def generate_report_prompt(self, prompt):
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

    def extract_report_score(self, text):
        score_match = re.search(r"\{.*\}", text)
        if score_match:
            score_text = score_match.group()
            score = ast.literal_eval(score_text)
            report = text[:score_match.start()].strip()
            return report, score['Report Quality']
        else:
            return text, 0

    def get_state(self, stage):
        """Define o estado atual baseado na última recompensa."""
        if stage == 'CODE':
            return self.programmer_reward_history[-1] if self.programmer_reward_history else 0
        elif stage == 'REVIEW':
            return self.reviewer_reward_history[-1] if self.reviewer_reward_history else 0
        return 0

    def explore(self, stage, state):
        # Seleciona uma ação (prompt) aleatória
        return self.prompt_master.get_random_action(stage)

    def exploit(self, stage, state):
        # Seleciona a melhor ação conhecida para o estado atual
        if state in self.q_table and self.q_table[state]:
            action = max(self.q_table[state], key=self.q_table[state].get)
            return action
        else:
            return self.explore(stage, state)

    def generate_hint(self):
        # Implementação para gerar uma dica (hint) usando PromptMaster
        # Placeholder: substitua com chamada real
        hint = "Foque em melhorar a clareza das suas revisões adicionando exemplos específicos."
        hint_strength = 75
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
        with open("data/reviewer_hints.txt", "w") as file:
            file.write(text)
        logging.info("Dicas e pesos do revisor armazenados.")

    def load_hints(self):
        try:
            with open("data/reviewer_hints.txt", "r") as file:
                lines = file.readlines()
            if lines:
                self.weights = ast.literal_eval(lines[0].strip())
                self.hints = "".join(lines[1:]).strip()
        except (SyntaxError, ValueError, FileNotFoundError) as e:
            logging.error(f"Erro ao carregar dicas do revisor: {e}")
            self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
            self.hints = ""

    def reset(self):
        # Resetar histórico após um ciclo de treinamento
        self.prompt_history = []
        self.review_history = []
        self.report_history = []
        self.reward_history = []
    
    def save(self, filepath):
        """Salva o estado atual do agente Reviewer em um arquivo."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Revisor salvo em {filepath}")

    @staticmethod
    def load(filepath):
        """Carrega o estado do agente Reviewer a partir de um arquivo."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                reviewer = pickle.load(f)
            logging.info(f"Revisor carregado de {filepath}")
            return reviewer
        else:
            logging.warning(f"Arquivo {filepath} não encontrado. Inicializando um novo Revisor.")
            return None
