from ollama import Client
from programmer import programmer

client = Client(host='http://localhost:11434')

class reviewer():

    def __init__(self):
        self.prompt = ("You are an expert code reviewer tasked with performing a comprehensive review. Your review should cover all critical aspects, "
            "including code clarity, readability, efficiency and optimization, also check for proper use of imports and adherence to best practices. "
            "Your feedback should be detailed and strict, addressing potential improvements, and noting any issues with syntax, style, or logic. "
            "You must give the code a score from 1 to 100, in all the critical aspects previously mentioned, be strict when scoring."
            " In the END of your response add in this format the scores you came up:\n"
            "{'Total':score, 'clarity':score, 'readability':score, 'efficiency':score, 'optimization':score}"
            " This MUST be in the end of the response, and NOTHING must be after it.")
        
        self.current_prompt = ""
        self.hints = ""
        self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}
        self.prompt_history = []
        self.review_history = []
        self.reward_history = []
        self.max_attempts = 10

    def add_hint(self, new_hint, hint_weight):
        self.hints += "\n" + "- " + str(new_hint) + f"(Weight: {hint_weight})"
        
    def _set_current_prompt(self, code):
        self.current_prompt = self.prompt + "\n The total score must be a weighted average of the other taking the following weights:"
        self.current_prompt += str(self.weights)
        self.current_prompt += "Please take the following hints into consideration, each with a weight from 1 to 100, indicating their importance. The hints are:"
        self.current_prompt += self.hints
        self.current_prompt += "\n Now, following all the previous rules, review the code:\n"
        self.current_prompt += code

        self.prompt_history.append(self.current_prompt)

    def code(self, code):
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

        review = response['message']['content']

        self.review_history.append(review)

        return review

if __name__ == '__main__':
    
    reviewer = reviewer()
    programmer = programmer()

    code = programmer.code('Given an array of integers nums and an integer target,\
                     return indices of the two numbers such that they add up to target.\
                     You may assume that each input would have exactly one solution,\
                     and you may not use the same element twice. You can return the answer in any order.')

    review = reviewer.code(code)
    
    print(review)




