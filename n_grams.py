import math
from collections import Counter, defaultdict

# Unigram Model
class UnigramModel:
    def __init__(self):
        self.token_counts = Counter()
        self.total_tokens = 0

    def train(self, sentences):
        for sentence in sentences:
            self.token_counts.update(sentence)
            self.total_tokens += len(sentence)

    def probability(self, token):
        if token in self.token_counts:
            return self.token_counts[token] / self.total_tokens
        else:
            return 0
        
# Bigram Model
class BigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()

    def train(self, sentences):
        for sentence in sentences:
            prev_token = None
            for token in sentence:
                if prev_token is not None:
                    self.bigram_counts[prev_token][token] += 1
                self.unigram_counts[prev_token] += 1
                prev_token = token

    def probability(self, token, prev_token):
        if prev_token in self.bigram_counts and token in self.bigram_counts[prev_token]:
            return self.bigram_counts[prev_token][token] / self.unigram_counts[prev_token]
        else:
            return 0

# Trigram Model
class TrigramModel:
    def __init__(self):
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        self.bigram_counts = defaultdict(Counter)

    def train(self, sentences):
        for sentence in sentences:
            prev_token_1, prev_token_2 = None, None
            for token in sentence:
                if prev_token_1 is not None and prev_token_2 is not None:
                    self.trigram_counts[prev_token_1][prev_token_2][token] += 1
                if prev_token_1 is not None:
                    self.bigram_counts[prev_token_1][prev_token_2] += 1
                prev_token_1, prev_token_2 = prev_token_2, token

    def probability(self, token, prev_token_1, prev_token_2):
        if (prev_token_1 in self.trigram_counts 
            and prev_token_2 in self.trigram_counts[prev_token_1] 
            and token in self.trigram_counts[prev_token_1][prev_token_2]):
            trigram_count = self.trigram_counts[prev_token_1][prev_token_2][token]
            bigram_count = self.bigram_counts[prev_token_1][prev_token_2]
            return trigram_count / bigram_count if bigram_count > 0 else 0
        else:
            return 0 

def calculate_perplexity(model, sentences, model_type='unigram'):
    total_log_prob = 0
    total_tokens = 0
    
    for sentence in sentences:
        for i in range(len(sentence)):
            token = sentence[i]

            if model_type == 'unigram':
                prob = model.probability(token)
            elif model_type == 'bigram':
                prev_token = sentence[i - 1] if i > 0 else '<START>'
                prob = model.probability(token, prev_token)
            elif model_type == 'trigram':
                prev_token_1 = sentence[i - 2] if i > 1 else '<START>'
                prev_token_2 = sentence[i - 1] if i > 0 else '<START>'
                prob = model.probability(token, prev_token_1, prev_token_2)

            if prob > 0:
                total_log_prob += math.log2(prob)
                total_tokens += 1
            
    if total_tokens == 0:
        return float('inf')  # Perplexity is infinite if no tokens have non-zero probability

    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.pow(2, -avg_log_prob)
    return perplexity

# Preprocess data
def preprocess_data(file_path, min_freq=3):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sentences = []
    for line in lines:
        tokens = line.strip().split() 
        tokens = ['<START>'] + tokens + ['<STOP>'] 
        sentences.append(tokens)
    
    # Flatten
    all_tokens = [token for sentence in sentences for token in sentence]

    # Count the occurrences of each token
    token_counts = Counter(all_tokens)

    # Create a list of rare words that appear less than or equal to min_freq times
    rare_words = {word for word, count in token_counts.items() if count <= min_freq}

    # Replace rare words with <UNK> in the sentences
    for i, sentence in enumerate(sentences):
        sentences[i] = [token if token not in rare_words else '<UNK>' for token in sentence]
    
    # Remove <START> token from all sentences
    # sentences = [[token for token in sentence if token != '<START>'] for sentence in sentences]
    
    # Return the processed sentences
    return sentences

def tokenize_test_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sentences = []
    for line in lines:
        tokens = line.strip().split()  # Just split the sentence into tokens
        tokens = ['<START>'] + tokens + ['<STOP>']  # Add start and stop tokens
        sentences.append(tokens)

    return sentences

# Main function to run everything
def main():
    # Preprocess data
    train_data = preprocess_data('1b_benchmark.train.tokens')


    # Added these print statements to test how preprocessing works
    all_tokens = [token for sentence in train_data for token in sentence]
    print("Number of tokens in train set:", len(all_tokens))
    unique_tokens = set(all_tokens)
    print("Number of unique tokens in train set:", len(unique_tokens))
    # Print the first 5 sentences of the train set
    for sentence in train_data[:5]:
        print(" ".join(sentence))  # Join the tokens into a single string for readability



    #dev_data  = preprocess_data('1b_benchmark.dev.tokens')
    # test_data  = preprocess_data('1b_benchmark.test.tokens')
    test = tokenize_test_data('test.tokens')
    print(test)

    # Train models
    unigram_model = UnigramModel()
    bigram_model = BigramModel()
    trigram_model = TrigramModel()

    unigram_model.train(train_data)
    bigram_model.train(train_data)
    trigram_model.train(train_data)

    # Calculate perplexity for train set
    print("Unigram Perplexity on Train Set:", calculate_perplexity(unigram_model, train_data, 'unigram'))
    print("Bigram Perplexity on Train Set:", calculate_perplexity(bigram_model, train_data, 'bigram'))
    print("Trigram Perplexity on Train Set:", calculate_perplexity(trigram_model, train_data, 'trigram'))
    print()
    # # Calculate perplexity for test HDTV
    print("Unigram Perplexity on Test Set:", calculate_perplexity(unigram_model, test, 'unigram'))
    print("Bigram Perplexity on Test Set:", calculate_perplexity(bigram_model, test, 'bigram'))
    print("Trigram Perplexity on Test Set:", calculate_perplexity(trigram_model, test, 'trigram'))

    # DONT UNCOMMENT THESE, LETS JUST FOCUS ON THE TRAIN AND THE HDTV TEST
    # Calculate perplexity for dev set
    # print("Unigram Perplexity on Dev Set:", calculate_perplexity(unigram_model, dev_data, 'unigram'))
    # print("Bigram Perplexity on Dev Set:", calculate_perplexity(bigram_model, dev_data, 'bigram'))
    # print("Trigram Perplexity on Dev Set:", calculate_perplexity(trigram_model, dev_data, 'trigram'))

    # Calculate perplexity for test set
    # print("Unigram Perplexity on Test Set:", calculate_perplexity(unigram_model, test_data, 'unigram'))
    # print("Bigram Perplexity on Test Set:", calculate_perplexity(bigram_model, test_data, 'bigram'))
    # print("Trigram Perplexity on Test Set:", calculate_perplexity(trigram_model, test_data, 'trigram'))

# Run the main function
if __name__ == "__main__":
    main()
