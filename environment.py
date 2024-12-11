from programmer import Programmer
from reviewer import Reviewer
from prompt_master import PromptMaster

import matplotlib.pyplot as plt
import os
import subprocess
import re
import time
import json
import logging

class Environment:
    def __init__(self, programmer: Programmer, reviewer: Reviewer, prompt_master: PromptMaster):
        self.programmer = programmer
        self.reviewer = reviewer
        self.prompt_master = prompt_master
        self.code_scores = []
        self.report_scores = []
        self.programmer_q_values_history = []
        self.reviewer_q_values_history = []


    def eval_code(self, code: str):
        code = re.sub(r"```python|```", "", code).strip()

        with open("temp_code.py", "w") as temp_file:
            temp_file.write(code)

        max_score = 3
        score = 3

        # Mypy
        mypy_result = subprocess.run(["mypy", "temp_code.py"], capture_output=True, text=True)
        if "error" in mypy_result.stdout.lower():
            score -= 1

        # Ruff
        ruff_result = subprocess.run(["ruff", "temp_code.py"], capture_output=True, text=True)
        ruff_issues = ruff_result.stdout.count("E") + ruff_result.stdout.count("W")
        if ruff_issues > 0:
            score -= min(1.0, ruff_issues * 0.1)

        # Bandit
        bandit_result = subprocess.run(["bandit", "-r", "temp_code.py"], capture_output=True, text=True)
        bandit_issues = bandit_result.stdout.count("Issue")
        if bandit_issues > 0:
            score -= min(1.0, bandit_issues * 0.2)

        os.remove("temp_code.py")

        try:
            with open("temp_code.py", "w") as temp_file:
                temp_file.write(code)
            subprocess.check_output(["python", "temp_code.py"], stderr=subprocess.STDOUT)
            score = 1.0
        except subprocess.CalledProcessError as e:
            logging.error(f"Erro na execução do código: {e.output}")
            score = 0.0
        except Exception as e:
            logging.error(f"Erro inesperado durante a avaliação do código: {e}")
            score = 0.0
        finally:
            if os.path.exists("temp_code.py"):
                os.remove("temp_code.py")
        return score

    def run_code(self, code: str):
        code = re.sub(r"```python|```", "", code).strip()

        temp_file_path = "temp_code.py"

        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(code)

        try:
            start_time = time.time()

            result = subprocess.run(
                ["python", temp_file_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=100
            )

            end_time = time.time()
            execution_time = end_time - start_time

            if result.returncode == 0:
                # Success
                return True, execution_time, result.stdout.strip()
            else:
                # Fail
                return False, execution_time, result.stderr.strip()

        except Exception as e:
            return False, 0, str(e)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def calculate_reward(self, code_score, report_score):
        # Calcula a recompensa baseada na pontuação do código e do relatório
        normalized_code_score = code_score  # Já está entre 0 e 1
        normalized_report_score = report_score / 100  # Normaliza para [0,1]
        total_reward = (normalized_code_score + normalized_report_score) / 2
        return total_reward

    def train(self, problem):
        question = problem["question"]
        data = problem["data"]
        metrics = problem["metrics"]

        logging.info(f"\n--- Treinando com o problema: {question} ---")

        # Passo 1: Agente Codificador gera o código
        code = self.programmer.act(question, training=True)
        logging.info(f"Código Gerado:\n{code}")

        # Passo 2: Executar e avaliar o código
        success, exec_time, output = self.run_code(code)
        code_score = self.eval_code(code)
        logging.info(f"Execução do Código: {'Sucesso' if success else 'Falha'}, Tempo: {exec_time}, Output: {output}")
        logging.info(f"Pontuação do Código: {code_score}")

        # Passo 3: Agente Revisor revisa o código
        action, review, review_score = self.reviewer.act(code, training=True)
        logging.info(f"Revisão:\n{review}")
        logging.info(f"Pontuação da Revisão: {review_score}")

        # Passo 4: Gerar e avaliar o relatório
        report, report_score = self.reviewer.generate_report(code)
        logging.info(f"Relatório:\n{report}")
        logging.info(f"Pontuação do Relatório: {report_score}")

        # Passo 5: Calcular recompensa
        reward = self.calculate_reward(code_score, report_score)
        logging.info(f"Recompensa Calculada: {reward}")

        # Passo 6: Atualizar políticas dos agentes com base na recompensa
        self.programmer.update_policy(state=self.programmer.get_state(), action=code, reward=reward)
        self.reviewer.update_policy(state=self.reviewer.get_state(stage='REVIEW'), action=action, reward=reward)

        # Armazenar valores para plotagem
        self.code_scores.append(code_score)
        self.report_scores.append(report_score)
        self.programmer_q_values_history.append(self.programmer.get_average_q_value())
        self.reviewer_q_values_history.append(self.reviewer.get_average_q_value())

    def test(self, problem):
        question = problem["question"]
        data = problem["data"]
        metrics = problem["metrics"]

        logging.info(f"\n--- Testando com o problema: {question} ---")

        # Passo 1: Agente Codificador gera o código
        code = self.programmer.act(question, training=False)
        logging.info(f"Código Gerado:\n{code}")

        # Passo 2: Executar e avaliar o código
        success, exec_time, output = self.run_code(code)
        code_score = self.eval_code(code)
        logging.info(f"Execução do Código: {'Sucesso' if success else 'Falha'}, Tempo: {exec_time}, Output: {output}")
        logging.info(f"Pontuação do Código: {code_score}")

        # Passo 3: Agente Revisor revisa o código
        # Aqui precisamos obter (action, review, review_score) mesmo no teste
        action, review, review_score = self.reviewer.act(code, training=False)
        logging.info(f"Revisão:\n{review}")
        logging.info(f"Pontuação da Revisão: {review_score}")

        # Passo 4: Gerar e avaliar o relatório
        report, report_score = self.reviewer.generate_report(code)
        logging.info(f"Relatório:\n{report}")
        logging.info(f"Pontuação do Relatório: {report_score}")

        # Calcular recompensa (poderíamos não precisar nessa fase, mas mantemos)
        reward = self.calculate_reward(code_score, report_score)
        logging.info(f"Recompensa Calculada: {reward}")

        # Armazenar os valores atuais nas listas de histórico
        self.code_scores.append(code_score)
        self.report_scores.append(report_score)
        self.programmer_q_values_history.append(self.programmer.get_average_q_value())
        self.reviewer_q_values_history.append(self.reviewer.get_average_q_value())

        # Após o teste, gerar gráficos
        self.plot_results()

    def plot_results(self):
        # Gera gráficos a partir das listas code_scores, report_scores, programmer_q_values_history e reviewer_q_values_history

        # 1. Gráfico de pontuação do código ao longo do tempo
        plt.figure(figsize=(10,6))
        plt.plot(self.code_scores, label='Pontuação do Código')
        plt.xlabel('Iterações')
        plt.ylabel('Pontuação do Código (0 a 1)')
        plt.title('Evolução da Pontuação do Código')
        plt.legend()
        plt.savefig('evolucao_codigo.png')
        plt.close()

        # 2. Gráfico de pontuação do relatório ao longo do tempo
        plt.figure(figsize=(10,6))
        plt.plot(self.report_scores, label='Pontuação do Relatório')
        plt.xlabel('Iterações')
        plt.ylabel('Pontuação do Relatório (0 a 100)')
        plt.title('Evolução da Pontuação do Relatório')
        plt.legend()
        plt.savefig('evolucao_relatorio.png')
        plt.close()

        # 3. Gráfico da média da Q-table do Programmer
        plt.figure(figsize=(10,6))
        plt.plot(self.programmer_q_values_history, label='Média Q-table Programmer')
        plt.xlabel('Iterações')
        plt.ylabel('Média Q-table')
        plt.title('Evolução da Média da Q-table do Programmer')
        plt.legend()
        plt.savefig('evolucao_q_programmer.png')
        plt.close()

        # 4. Gráfico da média da Q-table do Reviewer
        plt.figure(figsize=(10,6))
        plt.plot(self.reviewer_q_values_history, label='Média Q-table Reviewer')
        plt.xlabel('Iterações')
        plt.ylabel('Média Q-table')
        plt.title('Evolução da Média da Q-table do Reviewer')
        plt.legend()
        plt.savefig('evolucao_q_reviewer.png')
        plt.close()

        logging.info("Gráficos salvos: evolucao_codigo.png, evolucao_relatorio.png, evolucao_q_programmer.png, evolucao_q_reviewer.png")

    def create_test_cases(self, question):
        """Create test cases based on the question."""
        # This function should generate test inputs and expected outputs (gabarito)
        # For demonstration, we'll define them manually
        if "two numbers such that they add up to target" in question.lower():
            test_cases = [
                {
                    "nums": [2, 7, 11, 15],
                    "target": 9,
                    "expected": [0, 1]
                },
                {
                    "nums": [3, 2, 4],
                    "target": 6,
                    "expected": [1, 2]
                },
                {
                    "nums": [3, 3],
                    "target": 6,
                    "expected": [0, 1]
                }
            ]
            return test_cases
        # Add more problem types as needed
        else:
            return []

    def evaluate_tests(self, code: str, test_cases: list):
        """Evaluate the code against the test cases."""
        # Save the code to a temporary file
        code = re.sub(r"```python|```", "", code).strip()
        temp_file_path = "temp_code.py"

        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(code)

        all_passed = True
        for test in test_cases:
            nums = test['nums']
            target = test['target']
            expected = test['expected']

            try:
                result = subprocess.run(
                    ["python", temp_file_path],
                    input=json.dumps({"nums": nums, "target": target}),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    all_passed = False
                    break
                output = json.loads(result.stdout)
                if sorted(output) != sorted(expected):
                    all_passed = False
                    break
            except Exception as e:
                all_passed = False
                break

        os.remove(temp_file_path)
        return all_passed

if __name__ == '__main__':

    programmer = Programmer()
    reviewer = Reviewer()
    promptMaster = PromptMaster()

    env = Environment(programmer, reviewer, promptMaster)

    question = 'Given an array of integers nums and an integer target,\
                return indices of the two numbers such that they add up to target.\
                You may assume that each input would have exactly one solution,\
                and you may not use the same element twice. You can return the answer in any order.'

    env.train(question, 10)
    env.test(question)
