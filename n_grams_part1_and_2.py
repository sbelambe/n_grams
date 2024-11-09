import math
from collections import defaultdict, Counter


class NGramModel:
    """ Base class for n-gram models. """
    def __init__(self, n, train_data):
        self.n = n
        self.train_data = train_data
        self.ngram_counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)
        self.vocab = Counter()
        self.total_tokens = 0

    def preprocess(self):
        """ Preprocess the training data to build the vocabulary. """
        for sentence in self.train_data:
            tokens = ['<START>'] * (self.n - 1) + sentence.split() + ['<STOP>']
            for token in tokens:
                self.vocab[token] += 1
            self.total_tokens += len(tokens)

        self.vocab = {word: count for word, count in self.vocab.items() if count >= 3}
        self.vocab['<UNK>'] = 0 

    def count_ngrams(self):
        """ Count n-grams in the training data. """
        for sentence in self.train_data:
            tokens = ['<START>'] * (self.n - 1) + sentence.split() + ['<STOP>']
            tokens = [token if token in self.vocab else '<UNK>' for token in tokens]

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[context][word] += 1
                self.context_totals[context] += 1 

    def get_probability(self, context, word):
        """ Get the probability of a word given a context. """
        if context not in self.ngram_counts:
            return self.vocab.get('<UNK>', 0) / self.total_tokens
        context_count = self.context_totals[context]
        word_count = self.ngram_counts[context][word] if word in self.ngram_counts[context] else self.ngram_counts[context].get('<UNK>', 0)
        return word_count / context_count

    def perplexity(self, data):
        """ Calculate the perplexity of the model on a dataset. """
        log_prob_sum = 0
        token_count = 0
        sentence_count = 0 

        for sentence in data:
            tokens = ['<START>'] * (self.n - 1) + sentence.split() + ['<STOP>']
            tokens = [token if token in self.vocab else '<UNK>' for token in tokens]

            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]
                prob = self.get_probability(context, word)

                if prob == 0:
                    print(f"Zero probability encountered at context {context} with word '{word}'.")
                    return float('inf')

                log_prob_sum += math.log(prob, 2)
                token_count += 1

        return math.pow(2, -log_prob_sum / token_count)

class UnigramModel(NGramModel):
    """ Class for unigram models. """
    def __init__(self, train_data):
        super().__init__(1, train_data)
        self.preprocess()
        self.count_ngrams()

class BigramModel(NGramModel):
    """ Class for bigram models. """
    def __init__(self, train_data):
        super().__init__(2, train_data)
        self.preprocess()
        self.count_ngrams()

class TrigramModel(NGramModel):
    """ Class for trigram models. """
    def __init__(self, train_data):
        super().__init__(3, train_data)
        self.preprocess()
        self.count_ngrams()

def load_data(filepath):
    """ Load data from a file. """
    with open(filepath, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

if __name__ == "__main__":
    train_data = load_data("1b_benchmark.train.tokens")
    dev_data = load_data("1b_benchmark.dev.tokens")
    test_data = load_data("1b_benchmark.test.tokens")
    hdtv_test_data = ["HDTV ."]

    unigram_model = UnigramModel(train_data)
    bigram_model = BigramModel(train_data)
    trigram_model = TrigramModel(train_data)

    print("Number of tokens in train set:", unigram_model.total_tokens)
    print("Number of unique tokens in train set:", len(unigram_model.vocab))

    print("Unigram Perplexity on Train Set:", unigram_model.perplexity(train_data))
    print("Bigram Perplexity on Train Set:", bigram_model.perplexity(train_data))
    print("Trigram Perplexity on Train Set:", trigram_model.perplexity(train_data))

    print("Unigram Perplexity on Dev Set:", unigram_model.perplexity(dev_data))
    print("Bigram Perplexity on Dev Set:", bigram_model.perplexity(dev_data))
    print("Trigram Perplexity on Dev Set:", trigram_model.perplexity(dev_data))

    print("Unigram Perplexity on Test Set:", unigram_model.perplexity(test_data))
    print("Bigram Perplexity on Test Set:", bigram_model.perplexity(test_data))
    print("Trigram Perplexity on Test Set:", trigram_model.perplexity(test_data))

    print("Unigram Perplexity on HDTV Test Set:", unigram_model.perplexity(hdtv_test_data))
    print("Bigram Perplexity on HDTV Test Set:", bigram_model.perplexity(hdtv_test_data))
    print("Trigram Perplexity on HDTV Test Set:", trigram_model.perplexity(hdtv_test_data))
