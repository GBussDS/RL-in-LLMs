from ollama import Client
from programmer import Programmer
import ast
import re

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
            "No additional text, explanations, or commentary should appear after this line, as it will not be considered. Do not write anything after the scores.")

        
        self.current_prompt = ""
        
        if new:
            self.hints = ""
            self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
        else:
            self.hints, self.weights = self._get_hints()

        self.prompt_history = []
        self.review_history = []
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
        response ={'done_reason':None}
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
    
    def _extract_score(self, text):
        lines = text.strip().split('\n')

        score_match = re.search(r"\{.*\}", text)
        score_text = score_match.group()

        score = ast.literal_eval(score_text)

        review = text[:score_match.start()].strip()

        return review, score

    def update(self, new_hint, hint_weight, weights):
        if new_hint != None:
            if hint_weight != None:
                self.hints += "\n" + "- " + str(new_hint) + f"(Weight: {hint_weight})"
            else:
                self.hints += "\n" + "- " + str(new_hint) + f"(Weight: 70)"

        if weights != None:
            self.weights = weights
        
        self._store_hints()
    
    def _store_hints(self):
        text = str(self.weights) + self.hints
        with open("data/reviewer_hints.txt", "w") as file:
            file.write(text)
    
    def _get_hints(self):
        with open("data/reviewer_hints.txt", "r") as file:
            lines = file.readline()
        
        if len(lines) > 0:
            hints = lines[0].strip()
            weights = "".join(lines[1:]).strip()
        else:
            hints = ""
            weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}

        return hints, weights

if __name__ == '__main__':
    
    reviewer = Reviewer()
    programmer = Programmer()

    code = programmer.code('Given an array of integers nums and an integer target,\
                     return indices of the two numbers such that they add up to target.\
                     You may assume that each input would have exactly one solution,\
                     and you may not use the same element twice. You can return the answer in any order.')

    review = reviewer.review(code)
    
    print(review)




