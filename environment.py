from ollama import Client
from programmer import Programmer
from reviewer import Reviewer
from prompt_master import PromptMaster

import os
import subprocess
from typing import Optional
import re

class Environment():
    def __init__(self, programmer: Programmer, reviewer: Reviewer, promptMaster: PromptMaster):
        self.programmer = programmer
        self.reviewer = reviewer
        self.promptMaster = promptMaster

    def eval_code(self, code: str):
        code = re.sub(r"```python|```", "", code).strip()

        with open("temp_code.py", "w") as temp_file:
            temp_file.write(code)

        max_score = 3
        score = 3

        #Mypy
        mypy_result = subprocess.run(["mypy", "temp_code.py"], capture_output=True, text=True)
        if "error" in mypy_result.stdout.lower():
            score -= 1

        #Ruff
        ruff_result = subprocess.run(["ruff", "temp_code.py"], capture_output=True, text=True)
        ruff_issues = ruff_result.stdout.count("E") + ruff_result.stdout.count("W")
        if ruff_issues > 0:
            score -= min(1.0, ruff_issues * 0.1)

        # Step 4: Bandit security check
        bandit_result = subprocess.run(["bandit", "-r", "temp_code.py"], capture_output=True, text=True)
        bandit_issues = bandit_result.stdout.count("Issue")
        if bandit_issues > 0:
            score -= min(1.0, bandit_issues * 0.2)

        os.remove("temp_code.py")

        final_score = score / max_score

        return final_score


    def train(self, question, iterations):
        for i in range(iterations):
            code = self.programmer.code(question)
            review, score = self.reviewer.review(code)
            code_hint, code_hint_stren, new_code_weight = self.promptMaster.create_hint('CODE', code, review, score, self.programmer.weights)

            self.programmer.update(code_hint, code_hint_stren, new_code_weight)

            code = self.programmer.code(question)
            review, score = self.reviewer.review(code)
            
            review_hint, review_hint_stren, new_review_weight = self.promptMaster.create_hint('REVIEW', code, review, score, self.reviewer.weights)

            self.reviewer.update(review_hint, review_hint_stren, new_review_weight)

            print(score)

    def test(self, question):
        code = self.programmer.code(question)
        review, score = self.reviewer.review(code)

        print(f"With the score of {score} the code written was:\n\n{code}")

if __name__ == '__main__':
    
    programmer = Programmer()
    reviewer = Reviewer()
    promptMaster = PromptMaster()

    env = Environment(programmer, reviewer, promptMaster)
    
    env.train('Given an array of integers nums and an integer target,\
                return indices of the two numbers such that they add up to target.\
                You may assume that each input would have exactly one solution,\
                and you may not use the same element twice. You can return the answer in any order.', 10)
    
    env.test('Given an array of integers nums and an integer target,\
                return indices of the two numbers such that they add up to target.\
                You may assume that each input would have exactly one solution,\
                and you may not use the same element twice. You can return the answer in any order.')
    
