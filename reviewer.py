# reviewer.py

from ollama import Client
from programmer import Programmer
import ast
import re
import logging

client = Client(host='http://localhost:11434')

class Reviewer():

    def __init__(self, new=False):
        self.prompt = (
            "You are an expert code reviewer tasked with performing a comprehensive review. Your review should cover all critical aspects, "
            "including code clarity, readability, efficiency, and optimization, also check for proper use of imports and adherence to best practices. "
            "Your feedback should be detailed and strict, addressing potential improvements, and noting any issues with syntax, style, or logic. "
            "You must give the code a score from 1 to 100 in all the critical aspects previously mentioned. Be strict when scoring."
            " At the VERY END of your response, in the LAST line, you MUST add ONLY the scores in this exact format:\n"
            "{'Total':score, 'clarity':score, 'readability':score, 'efficiency':score, 'optimization':score}\n"
            "No additional text, explanations, or commentary should appear after this line, as it will not be considered. Do not write anything after the scores."
        )

        self.report_prompt = (
            "You are an expert data analyst tasked with creating a comprehensive data analysis report based on the provided code. "
            "The report should include data cleaning steps, transformations applied, insights derived, and any visualizations generated. "
            "Ensure the report is clear, concise, and well-structured, suitable for stakeholders to understand the data workflow and findings. "
            "At the VERY END of your report, in the LAST line, you MUST add ONLY the report's overall quality score from 1 to 100 in the following format:\n"
            "{'Report Quality': score}\n"
            "No additional text, explanations, or commentary should appear after this line, as it will not be considered."
        )

        self.current_prompt = ""

        if new:
            self.hints = ""
            self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
        else:
            self.hints, self.weights = self._get_hints()

        self.prompt_history = []
        self.review_history = []
        self.report_history = []
        self.reward_history = []
        self.max_attempts = 10

    def _set_current_prompt(self, code):
        self.current_prompt = self.prompt + "\n The total score must be a weighted average of the other taking the following weights:"
        self.current_prompt += str(self.weights)

        if self.hints != "":
            self.current_prompt += "Please take the following hints into consideration, each with a weight from 1 to 100, indicating their importance. The hints are:"
            self.current_prompt += self.hints

        self.current_prompt += "\n Now, following all the previous rules, review the code:\n"
        self.current_prompt += code

        self.prompt_history.append(self.current_prompt)

    def review(self, code):
        self._set_current_prompt(code)

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

        review, score = self._extract_score(response['message']['content'])

        self.review_history.append(review)

        return review, score

    def generate_report(self, code):
        """Generate a data analysis report based on the provided code."""
        attempts = 0
        response = {'done_reason': None}
        while response['done_reason'] != 'stop' and attempts < self.max_attempts:
            response = client.chat(model='llama3.1', messages=[
                {
                    'role': 'user',
                    'content': self.report_prompt + "\n\n" + code,
                },
            ])
            attempts += 1

        report, quality_score = self._extract_report_score(response['message']['content'])

        self.report_history.append(report)

        return report, quality_score

    def _extract_score(self, text):
        lines = text.strip().split('\n')
        score_match = re.search(r"\{.*\}", text)
        if score_match:
            score_text = score_match.group()
            score = ast.literal_eval(score_text)
            review = text[:score_match.start()].strip()
            return review, score
        else:
            return text, {'Total': 0, 'clarity': 0, 'readability': 0, 'efficiency': 0, 'optimization': 0}

    def _extract_report_score(self, text):
        score_match = re.search(r"\{.*\}", text)
        if score_match:
            score_text = score_match.group()
            score = ast.literal_eval(score_text)
            report = text[:score_match.start()].strip()
            return report, score['Report Quality']
        else:
            return text, 0

    def update(self, new_hint, hint_weight, weights):
        if new_hint is not None:
            if hint_weight is not None:
                self.hints += "\n- " + str(new_hint) + f" (Weight: {hint_weight})"
            else:
                self.hints += "\n- " + str(new_hint) + f" (Weight: 70)"

        if weights is not None:
            self.weights = weights

        self._store_hints()

    def _store_hints(self):
        # Ensure weights and hints are separated by a newline
        text = str(self.weights) + "\n" + self.hints
        with open("data/reviewer_hints.txt", "w") as file:
            file.write(text)
        logging.info("Stored reviewer hints and weights.")

    def _get_hints(self):
        try:
            with open("data/reviewer_hints.txt", "r") as file:
                lines = file.readlines()
            
            if lines:
                weights = ast.literal_eval(lines[0].strip())
                hints = "".join(lines[1:]).strip()
            else:
                weights = {'clarity': 1, 'readability': 1, 'efficiency': 1, 'optimization': 1}
                hints = ""
        except (SyntaxError, ValueError, FileNotFoundError) as e:
            logging.error(f"Error parsing reviewer hints: {e}")
            weights = {'clarity': 1, 'readability': 1, 'efficiency': 1, 'optimization': 1}
            hints = ""

        return hints, weights

if __name__ == '__main__':

    reviewer = Reviewer()
    programmer = Programmer()

    code = programmer.code('Given an array of integers nums and an integer target,\
                     return indices of the two numbers such that they add up to target.\
                     You may assume that each input would have exactly one solution,\
                     and you may not use the same element twice. You can return the answer in any order.')

    review, score = reviewer.review(code)
    report, report_score = reviewer.generate_report(code)

    print("Review:", review)
    print("Score:", score)
    print("Report:", report)
    print("Report Score:", report_score)
