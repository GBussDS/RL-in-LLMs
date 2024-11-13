from ollama import Client
from programmer import programmer
from reviewer import reviewer
from prompt_master import promptMaster

class main():
    def __init__(self, programmer:programmer, reviewer:reviewer, promptMaster:promptMaster):
        self.programmer = programmer
        self.reviewer = reviewer
        self.promptMaster = promptMaster

    def train(self, question, iterations):
        for i in range(iterations):
            code = self.programmer.code(question)
            review, score = self.reviewer.review(code)
            code_hint, code_hint_stren, new_code_weight = self.promptMaster.create_hint('CODE', code, review, score, programmer.weights)

            self.programmer.update(code_hint, code_hint_stren, new_code_weight)

            code = self.programmer.code(question)
            review, score = self.reviewer.review(code)
            print(score)
            review_hint, review_hint_stren, new_review_weight = self.promptMaster.create_hint('REVIEW', code, review, score, programmer.weights)

            self.programmer.update(review_hint, review_hint_stren, new_review_weight)


    def test(self, question):
        code = self.programmer.code(question)
        review, score = self.reviewer.review(code)

        print(f"With the score of {score} the code written was:\n\n{code}")

if __name__ == '__main__':
    
    programmer = programmer()
    reviewer = reviewer()
    promptMaster = promptMaster()

    main = main(programmer, reviewer, promptMaster)
    
    main.train('Given an array of integers nums and an integer target,\
                return indices of the two numbers such that they add up to target.\
                You may assume that each input would have exactly one solution,\
                and you may not use the same element twice. You can return the answer in any order.', 10)
    
    main.test('Given an array of integers nums and an integer target,\
                return indices of the two numbers such that they add up to target.\
                You may assume that each input would have exactly one solution,\
                and you may not use the same element twice. You can return the answer in any order.')
    
