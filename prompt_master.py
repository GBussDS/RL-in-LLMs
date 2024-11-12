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

                "\n\nCODE: You are an expert prompt master responsible for providing a general, long-lasting hint that will help the programmer iteratively improve their code, "
                "focusing on overall quality enhancements that will remain relevant across different versions and types of code. The hint should not reference any specific code content "
                "but should offer guidance that could improve clarity, readability, efficiency, or optimization. Provide only the hint and an emphasis score, which is a single number "
                "from 1 to 100, indicating how strongly the hint should be followed. Include the hint, the emphasis score, and the weights for each area in the following format, and "
                "ensure that NOTHING follows this format (make sure to write between <>):\n"
                "Hint: <Your hint here>\n"
                "Emphasis: <1-100>\n"
                "{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}"

                "\n\nREVIEW: You are an expert prompt master responsible for providing a general, long-lasting hint that will help the reviewer improve their feedback. "
                "Your hint should be applicable to multiple rounds of review and focus on enhancing clarity, readability, efficiency, or optimization in their feedback, "
                "without referencing any specific feedback or code instance. Provide only the hint and an emphasis score, which is a single number from 1 to 100, indicating how strongly "
                "the hint should be prioritized in their next review. Include the hint, the emphasis score, and the weights in the following format, ensuring NOTHING follows this format:\n"
                "Hint: <Your hint here>\n"
                "Emphasis: <1-100>\n"
                "{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}")


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

        hint, hint_strenght, weights = self._extract_info(response['message']['content'])

        return hint, hint_strenght, weights
    
    def _extract_info(self, text):
        lines = text.strip().split('\n')

        weights_text = re.search(r"\{.*\}", lines[-1]).group()
        weights = ast.literal_eval(weights_text)

        hint_text = "\n".join(lines[:-2])
        hint = re.search(r"<(.*?)>", hint_text).group(1)
        hint_strength = re.search(r"<(.*?)>", lines[-2]).group(1)
        
        return hint, hint_strength, weights

if __name__ == '__main__':
    
    reviewer = reviewer()
    programmer = programmer()
    promptMaster = promptMaster()

    code = programmer.code('Given an array of integers nums and an integer target,\
                     return indices of the two numbers such that they add up to target.\
                     You may assume that each input would have exactly one solution,\
                     and you may not use the same element twice. You can return the answer in any order.')

    review, score = reviewer.review(code)

    hint, hint_stren, new_weight = promptMaster.create_hint('CODE', code, review, score, programmer.weights)
    
    print(hint)
    print(new_weight)