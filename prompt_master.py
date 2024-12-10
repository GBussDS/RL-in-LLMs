# prompt_master.py

from ollama import Client
import ast
import re
import random
import logging

client = Client(host='http://localhost:11434')

class PromptMaster:
    def __init__(self, epsilon=0.1):
        self.prompt = (
            "\nPreste atenção na primeira palavra deste prompt, pois ela ditará seu papel e abordagem. Se o prompt começar com 'CODE', "
            "você está guiando o programador. Se começar com 'REVIEW', você está guiando o revisor. Aja de acordo com a primeira palavra e siga as instruções específicas abaixo."

            "\n\nCODE: Você é um mestre de prompts responsável por fornecer uma dica geral e duradoura que ajudará o programador a melhorar iterativamente seu código, "
            "focando em aprimoramentos de qualidade geral que permanecerão relevantes em diferentes versões e tipos de código. A dica não deve referenciar nenhum conteúdo específico do código "
            "mas deve oferecer orientação que possa melhorar a clareza, legibilidade, eficiência ou otimização. Forneça apenas a dica e uma pontuação de ênfase, que é um único número "
            "de 1 a 100, indicando quão fortemente a dica deve ser seguida. Inclua a dica, a pontuação de ênfase e os pesos para cada área no seguinte formato, e "
            "certifique-se de que NADA siga este formato (garanta que os <> estejam incluídos na resposta três vezes):"
            "Dica: <Sua dica aqui>\n"
            "Ênfase: <1-100>\n"
            "<{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}> #formato de dicionário python"

            "\n\nREVIEW: Você é um mestre de prompts responsável por fornecer uma dica geral e duradoura que ajudará o revisor a melhorar seu feedback. "
            "Sua dica deve ser aplicável a múltiplas rodadas de revisão e focar em aprimorar a clareza, legibilidade, eficiência ou otimização no feedback, "
            "sem referenciar nenhum feedback ou instância de código específico. Forneça apenas a dica e uma pontuação de ênfase, que é um único número de 1 a 100, indicando quão fortemente "
            "a dica deve ser priorizada na próxima revisão. Inclua a dica, a pontuação de ênfase e os pesos no seguinte formato, garantindo que NADA siga este formato (garanta que os <> estejam incluídos na resposta três vezes):"
            "Dica: <Sua dica aqui>\n"
            "Ênfase: <1-100>\n"
            "<{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}> #formato de dicionário python"
        )
        self.current_prompt = ""
        self.programmer_weights_history = []
        self.reviewer_weights_history = []
        self.programmer_reward_history = []
        self.reviewer_reward_history = []
        self.max_attempts = 10
        self.epsilon = epsilon  # Probabilidade de explorar novas dicas

        # Inicializar tabelas Q para ações
        self.action_values = {
            'CODE': {},
            'REVIEW': {}
        }

        # Configurar logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _set_current_prompt(self, stage, code, review):
        self.current_prompt = stage + self.prompt
        self.current_prompt += "\n\nO código é:\n\n" + code
        self.current_prompt += "\n\nA revisão foi:\n\n" + review

        self.current_prompt += "\n\nPara referência, estes são os pesos que o " + stage + " usou e a pontuação subsequente que alcançou:" 

        if stage == 'CODE':
            for reward, weight in zip(self.programmer_reward_history, self.programmer_weights_history):
                self.current_prompt += "\n" + "Pontuação: "+ str(reward) + " Pesos: " + str(weight)
        elif stage == 'REVIEW':
            for reward, weight in zip(self.reviewer_reward_history, self.reviewer_weights_history):
                self.current_prompt += "\n" + "Pontuação: "+ str(reward) + " Pesos: " + str(weight)

    def create_hint(self, stage, code, review, score, weights):
        if stage == 'CODE':
            self.programmer_reward_history.append(score)
            self.programmer_weights_history.append(weights)
        elif stage == 'REVIEW':
            self.reviewer_reward_history.append(score)
            self.reviewer_weights_history.append(weights)

        self._set_current_prompt(stage, code, review)

        response = self.generate_hint(stage)
        hint, hint_strength, weights = self.extract_info(response['message']['content'])

        if hint and hint_strength:
            action = f"Dica: {hint}"
            if action not in self.action_values[stage]:
                self.action_values[stage][action] = 0
            try:
                self.action_values[stage][action] += float(hint_strength)
                logging.info(f"Adicionado/Atualizado ação '{action}' com força {hint_strength} no estágio '{stage}'.")
            except ValueError:
                logging.error(f"Pontuação de força inválida '{hint_strength}' para a ação '{action}'. Deve ser um número.")

        else:
            logging.warning(f"Falha ao extrair dica ou força da resposta: {response['message']['content']}")

        return hint, hint_strength, weights

    def generate_hint(self, stage):
        attempts = 0
        response = {'done_reason': None}
        while response['done_reason'] != 'stop' and attempts < self.max_attempts:
            response = client.chat(model='llama3.1', messages=[
                {
                    'role': 'user',
                    'content': self.current_prompt,
                },
            ])
            attempts += 1
        return response

    def extract_info(self, text):
        important = re.findall(r"<(.*?)>", text)
        hint = important[0] if len(important) > 0 and important[0] else None
        hint_strength = important[1] if len(important) > 1 and important[1] else None
        weights = important[2] if len(important) > 2 and important[2] else None
        return hint, hint_strength, weights

    def get_random_action(self, stage):
        # Implementação para selecionar uma dica aleatória se necessário
        return f"Dica: Sempre valide os dados antes de processá-los."

    def extract_hint(self, action):
        # Extrai a dica e a força da ação selecionada
        match = re.match(r"Dica:\s*(.*?)", action)
        if match:
            hint = match.group(1)
            # Assumindo que a força está armazenada na ação_values
            return hint, 80, None # self.weights   # Placeholder: ajustar conforme necessário
        return None, None, None

    def evaluate_action(self, stage, action, reward):
        """Atualiza o valor da ação baseada na recompensa recebida."""
        if action != 'NEW_PROMPT':
            if action in self.action_values[stage]:
                self.action_values[stage][action] += reward
                logging.info(f"Atualizado ação '{action}' no estágio '{stage}' com recompensa {reward}. Novo valor: {self.action_values[stage][action]}")
            else:
                # Inicializa a ação com a recompensa se não existir
                self.action_values[stage][action] = reward
                logging.warning(f"Ação '{action}' não encontrada no estágio '{stage}'. Inicializada com recompensa {reward}.")

    def get_state(self, stage):
        """Define o estado atual baseado na última recompensa."""
        if stage == 'CODE':
            return self.programmer_reward_history[-1] if self.programmer_reward_history else 0
        elif stage == 'REVIEW':
            return self.reviewer_reward_history[-1] if self.reviewer_reward_history else 0
        return 0

    def reset_history(self):
        """Reseta os históricos após um episódio."""
        self.programmer_reward_history = []
        self.programmer_weights_history = []
        self.reviewer_reward_history = []
        self.reviewer_weights_history = []
        logging.info("Históricos do PromptMaster resetados.")
