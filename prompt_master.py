# prompt_master.py

from ollama import Client
from programmer import Programmer
from reviewer import Reviewer
import ast
import re
import random
import logging

client = Client(host='http://localhost:11434')

class PromptMaster():

    def __init__(self, epsilon=0.1):
        self.prompt = (
            "\nPay close attention to the first word of this prompt, as it will dictate your role and approach. If the prompt starts with 'CODE', "
            "you are guiding the programmer. If it starts with 'REVIEW', you are guiding the reviewer. Act according to the first word and follow the specific instructions below."

            "\n\nCODE: You are an expert prompt master responsible for providing a general, long-lasting hint that will help the programmer iteratively improve their code, "
            "focusing on overall quality enhancements that will remain relevant across different versions and types of code. The hint should not reference any specific code content "
            "but should offer guidance that could improve clarity, readability, efficiency, or optimization. Provide only the hint and an emphasis score, which is a single number "
            "from 1 to 100, indicating how strongly the hint should be followed. Include the hint, the emphasis score, and the weights for each area in the following format, and "
            "ensure that NOTHING follows this format (make sure to write between <>).\n"
            "Make SURE the ONLY thing you answer is in the format, the <> brackets MUST be included in the answer three times:"
            "Hint: <Your hint here>\n"
            "Emphasis: <1-100>\n"
            "<{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}> #python dictionary format"

            "\n\nREVIEW: You are an expert prompt master responsible for providing a general, long-lasting hint that will help the reviewer improve their feedback. "
            "Your hint should be applicable to multiple rounds of review and focus on enhancing clarity, readability, efficiency, or optimization in their feedback, "
            "without referencing any specific feedback or code instance. Provide only the hint and an emphasis score, which is a single number from 1 to 100, indicating how strongly "
            "the hint should be prioritized in their next review. Include the hint, the emphasis score, and the weights in the following format, ensuring NOTHING follows this format (make sure to write between <>):\n"
            "Make SURE the ONLY thing you answer is in the format, the <> brackets MUST be included in the answer three times::"
            "Hint: <Your hint here>\n"
            "Emphasis: <1-100>\n"
            "<{'clarity': weight, 'readability': weight, 'efficiency': weight, 'optimization': weight}> #python dictionary format"
        )

        self.current_prompt = ""
        self.programmer_weights_history = []
        self.reviewer_weights_history = []
        self.programmer_reward_history = []
        self.reviewer_reward_history = []
        self.max_attempts = 10
        self.epsilon = epsilon  # Probability to explore new prompts

        # Initialize action-value tables
        self.action_values = {
            'CODE': {},
            'REVIEW': {}
        }

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _set_current_prompt(self, stage, code, review):
        self.current_prompt = stage + self.prompt
        self.current_prompt += "\n\nThe code is:\n\n" + code
        self.current_prompt += "\n\nThe review was:\n\n" + review

        self.current_prompt += "\n\nFor reference, these are the weights the " + stage + "R used and the subsequent score it achieved:" 

        if stage == 'CODE':
            for reward, weight in zip(self.programmer_reward_history, self.programmer_weights_history):
                self.current_prompt += "\n" + "Score: "+ str(reward) + " Weights: " + str(weight)
        elif stage == 'REVIEW':
            for reward, weight in zip(self.reviewer_reward_history, self.reviewer_weights_history):
                self.current_prompt += "\n" + "Score: "+ str(reward) + " Weights: " + str(weight)

    def select_action(self, stage):
        """Select an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon or not self.action_values[stage]:
            # Explore: return a request for a new prompt
            return 'NEW_PROMPT'
        else:
            # Exploit: choose the prompt with the highest reward
            best_prompt = max(self.action_values[stage], key=self.action_values[stage].get)
            return best_prompt

    def create_hint(self, stage, code, review, score, weights):
        if stage == 'CODE':
            self.programmer_reward_history.append(score)
            self.programmer_weights_history.append(weights)
        elif stage == 'REVIEW':
            self.reviewer_reward_history.append(score)
            self.reviewer_weights_history.append(weights)

        self._set_current_prompt(stage, code, review)

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

        hint, hint_strength, weights = self._extract_info(response['message']['content'])

        if hint and hint_strength:
            action = f"Hint: {hint}"
            if action not in self.action_values[stage]:
                self.action_values[stage][action] = 0
            try:
                self.action_values[stage][action] += float(hint_strength)
                logging.info(f"Added/Updated action '{action}' with strength {hint_strength} in stage '{stage}'.")
            except ValueError:
                logging.error(f"Invalid hint_strength '{hint_strength}' for action '{action}'. Must be a number.")

        else:
            logging.warning(f"Failed to extract hint or hint_strength from response: {response['message']['content']}")

        return hint, hint_strength, weights

    def _extract_info(self, text):
        important = re.findall(r"<(.*?)>", text)

        hint = important[0] if len(important) > 0 and important[0] else None
        hint_strength = important[1] if len(important) > 1 and important[1] else None
        weights = important[2] if len(important) > 2 and important[2] else None

        return hint, hint_strength, weights

    def evaluate_action(self, stage, action, reward):
        """Update action-value based on the received reward."""
        if action != 'NEW_PROMPT':
            if action in self.action_values[stage]:
                self.action_values[stage][action] += reward
                logging.info(f"Updated action '{action}' in stage '{stage}' with reward {reward}. New value: {self.action_values[stage][action]}")
            else:
                # Initialize the action with the reward if it doesn't exist
                self.action_values[stage][action] = reward
                logging.warning(f"Action '{action}' not found in stage '{stage}'. Initialized with reward {reward}.")

    def reset_history(self):
        """Reset histories after an episode."""
        self.programmer_reward_history = []
        self.programmer_weights_history = []
        self.reviewer_reward_history = []
        self.reviewer_weights_history = []
        logging.info("Reset all histories in PromptMaster.")
