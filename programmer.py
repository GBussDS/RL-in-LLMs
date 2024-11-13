from ollama import Client

client = Client(host='http://localhost:11434')

class Programmer():

    def __init__(self):
        self.prompt = ("You are a skilled programmer. Write only the code, without any additional text, "
               "language labels, headers, or explanations, but you may add comments in the code for readability."
               "The code should include all necessary imports and should be executable as-is. If creating a function, provide an example of its usage at the end.")
        self.current_prompt = ""
        self.hints = ""
        self.prompt_history = []
        self.code_history = []
        self.reward_history = []
        self.max_attempts = 10
        self.weights = {'clarity':1, 'readability':1, 'efficiency':1, 'optimization':1}

    def update(self, new_hint, hint_weight, weights):
        if new_hint != None:
            if hint_weight != None:
                self.hints += "\n" + "- " + str(new_hint) + f"(Weight: {hint_weight})"
            else:
                self.hints += "\n" + "- " + str(new_hint) + f"(Weight: 70)"

        if weights != None:
            self.weights = weights
        
    def _set_current_prompt(self, question):
        self.current_prompt = self.prompt + "\n When writing the code you should take the following weights in consideration, as to what you should focus more:\n"
        self.current_prompt += str(self.weights)
        self.current_prompt += "\nPlease take the following hints into consideration, each with a weight from 1 to 100, indicating their importance. The hints are:" + self.hints
        self.current_prompt += "\nNow, following all the previous rules, write a code to answer the question:\n"
        self.current_prompt += question
        self.prompt_history.append(self.current_prompt)

    def code(self, question):
        self._set_current_prompt(question)
       
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

        code = response['message']['content']

        self.code_history.append(code)

        return code

if __name__ == '__main__':

    programmer = Programmer()
    
    code = programmer.code('Given an array of integers nums and an integer target,\
                     return indices of the two numbers such that they add up to target.\
                     You may assume that each input would have exactly one solution,\
                     and you may not use the same element twice. You can return the answer in any order.')
    
    print(code)

    print(programmer.current_prompt)



