

class Configuration():
    def __init__(self):
        self.model = 'gpt-4-0613'  # "gpt-3.5-turbo" or "gpt-4"
        self.system_role =  "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2021-09. Current date: 2023-09-09"
        self.temperature = 0.5
        self.nb_attempts = 2
        self.sample_size = 5
        self.dataset_used = 'dataset/sign_events_data_statements.csv'
        self.classification_file = 'results/zero-shot-classification-with-clues-reasoning.csv'
    
    def print_config(self):
        print(f"{20*'-'} config {20*'-'}")
        print(f'Model: {self.model}')  # 'gpt-4-0613'  # "gpt-3.5-turbo" or "gpt-4"
        print(f'system role str: {self.system_role}')
        print(f'temperature: {self.temperature}')
        print(f'number of attempts to get good JSON: {self.nb_attempts}')
        print(f'sample size: {self.sample_size}')
        print(f'data set used: {self.dataset_used}')
        print(f'classification file: {self.classification_file}')
        print(f"{20*'-'} ++++++ {20*'-'}")

if __name__ == "__main__":
    config = Configuration()
    config.print_config()

