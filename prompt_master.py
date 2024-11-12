from ollama import Client
from programmer import programmer
from reviewer import reviewer
import ast
import re

client = Client(host='http://localhost:11434')

class promptMaster():

    def __init__(self):
        self.prompt = ("\nPay close attention to the first word of this prompt, as it will dictate your role and approach. If the prompt starts with 'CODE', "
               "you are guiding the programmer. If it starts with 'REVIEW', you are guiding the reviewer. Act according to the first word and follow the specific instructions below."

               "\n\nCODE: You are an expert prompt master responsible for guiding a programmer through iterative code improvement based on reviewer feedback. "
               "You will receive both the programmer's initial code and the reviewer's feedback, and your task is to generate a specific hint that "
               "will help the programmer enhance clarity, readability, efficiency, or optimization of their code. Your hint should be actionable, clear, "
               "and directly address the areas identified in the review. This hint will then be provided to the programmer with the assigned weights in the following format "
               "to indicate how much emphasis the programmer should place on each aspect in the next iteration:\n"
               "{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}. "
               "This MUST appear at the end of your response, with NOTHING following it."

               "\n\nREVIEW: You are an expert prompt master responsible for guiding a reviewer to deliver precise and constructive feedback on a programmer's code. "
               "You will receive the programmer's updated code and the reviewer's latest feedback. Your task is to analyze the effectiveness of the last hint provided "
               "to the programmer and assess the reviewer's feedback quality. Based on this analysis, generate a hint that will help the reviewer focus on areas that need "
               "greater attention or detail, such as clarity, readability, efficiency, or optimization, in the next round of review. Your goal is to improve the reviewer's "
               "ability to deliver actionable, balanced feedback. Once you've provided the hint, assign weights in the following format that will guide the reviewer's focus on "
               "each aspect, influencing the weighted mean for the review's overall score:\n"
               "{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}. "
               "This MUST appear at the end of your response, with NOTHING following it (This is of utmost importance).")

        self.current_prompt = ""

        self.programmer_weights_history = []
        self.reviewer_weights_history = []

        self.programmer_reward_history = []
        self.reviewer_reward_history = []

        self.max_attempts = 10

    def _set_current_prompt(self, stage, code, review):
        self.current_prompt = stage + self.prompt
        self.current_prompt += "\n\nThe code is:\n\n" + code
        self.current_prompt += "\n\nThe review was:\n\n" + review

        self.current_prompt += "\n\nFor reference, these are the weights the " + stage + "R used and the subsequent score it achieved:" 

        if stage == 'CODE':
            for reward,weight in zip(self.programmer_reward_history,self.programmer_weights_history):
                self.current_prompt += "\n" + "Score: "+ str(reward) + "Weights: " + str(weight)
        elif stage == 'REVIEW':
            for reward,weight in zip(self.reviewer_reward_history,self.reviewer_weights_history):
                self.current_prompt += "\n" + "Score: "+ str(reward) + "Weights: " + str(weight)

    def create_hint(self, stage, code, review, score, weights):
        if stage == 'CODE':
            self.programmer_reward_history.append(score)
            self.programmer_weights_history.append(weights)

        elif stage == 'REVIEW':
            self.reviewer_reward_history.append(score)
            self.reviewer_weights_history.append(weights)

        self._set_current_prompt(stage, code, review)
       
        attempts = 0
        response = {'done_reason':None}
        while response['done_reason'] != 'stop' and attempts < self.max_attempts:
            response = client.chat(model='llama3.1', messages=[
                {
                    'role': 'user',
                    'content': self.current_prompt,
                },
            ])
            attempts += 1

        hint, weights = self._extract_weight(response['message']['content'])

        return hint, weights
    
    def _extract_weight(self, text):
        lines = text.strip().split('\n')

        last_line = lines[-1]
        weights_text = re.search(r"\{.*\}", last_line).group()

        weights = ast.literal_eval(weights_text)

        hint = '\n'.join(lines[:-1])
        
        return hint, weights

if __name__ == '__main__':
    
    reviewer = reviewer()
    programmer = programmer()
    promptMaster = promptMaster()

    code = programmer.code('Given an array of integers nums and an integer target,\
                     return indices of the two numbers such that they add up to target.\
                     You may assume that each input would have exactly one solution,\
                     and you may not use the same element twice. You can return the answer in any order.')

    review, score = reviewer.review(code)

    hint, new_weight = promptMaster.create_hint('CODE', code, review, score, programmer.weights)
    
    print(hint)
    print(new_weight)