import math
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n, train_data, alpha=1e-10):  # Set a small default alpha
        self.n = n
        self.alpha = alpha
        self.train_data = train_data
        self.ngram_counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)
        self.vocab = Counter()
        self.total_tokens = 0

    def preprocess(self):
        for sentence in self.train_data:
            tokens = ['<START>'] * (self.n - 1) + sentence.split() + ['<STOP>']
            for token in tokens:
                self.vocab[token] += 1
            self.total_tokens += len(tokens)

        self.vocab = {word: count for word, count in self.vocab.items() if count >= 3}
        self.vocab['<UNK>'] = self.vocab.get('<UNK>', 1)

    def count_ngrams(self):
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
        vocab_size = max(len(self.vocab), 1)
        context_count = self.context_totals[context] + self.alpha * vocab_size
        word_count = self.ngram_counts[context][word] + self.alpha

        return word_count / context_count


class InterpolatedNGramModel:
    def __init__(self, train_data, lambda1, lambda2, lambda3):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Initialize the individual n-gram models with a small alpha to avoid zero division
        self.unigram_model = NGramModel(1, train_data, alpha=1e-10)
        self.bigram_model = NGramModel(2, train_data, alpha=1e-10)
        self.trigram_model = NGramModel(3, train_data, alpha=1e-10)

        self.unigram_model.preprocess()
        self.bigram_model.preprocess()
        self.trigram_model.preprocess()
        
        self.unigram_model.count_ngrams()
        self.bigram_model.count_ngrams()
        self.trigram_model.count_ngrams()

    def get_interpolated_probability(self, context, word):
        unigram_prob = self.unigram_model.get_probability((), word)
        bigram_prob = self.bigram_model.get_probability((context[-1],), word) if len(context) >= 1 else 0
        trigram_prob = self.trigram_model.get_probability(context[-2:], word) if len(context) >= 2 else 0

        return (self.lambda1 * unigram_prob +
                self.lambda2 * bigram_prob +
                self.lambda3 * trigram_prob)

    def perplexity(self, data):
        log_prob_sum = 0
        token_count = 0

        for sentence in data:
            tokens = ['<START>', '<START>'] + sentence.split() + ['<STOP>']
            tokens = [token if token in self.unigram_model.vocab else '<UNK>' for token in tokens]

            for i in range(2, len(tokens)):
                context = tuple(tokens[i-2:i])
                word = tokens[i]
                prob = self.get_interpolated_probability(context, word)
                log_prob_sum += math.log(prob, 2)
                token_count += 1

        return math.pow(2, -log_prob_sum / token_count)


def load_data(filepath):
    with open(filepath, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]


if __name__ == "__main__":
    # Load datasets
    train_data = load_data("1b_benchmark.train.tokens")
    dev_data = load_data("1b_benchmark.dev.tokens")
    test_data = load_data("1b_benchmark.test.tokens")
    hdtv_test_data = ["HDTV ."]

    # Lambda values to experiment with
    lambda_sets = [
        (0.1, 0.3, 0.6),
        (0.3, 0.3, 0.4),
        (0.2, 0.4, 0.4),
        (0.4, 0.3, 0.3),
        (0.5, 0.3, 0.2)
    ]

    # Evaluate on training and development sets
    for lambdas in lambda_sets:
        lambda1, lambda2, lambda3 = lambdas
        print(f"\nLinear Interpolation with λ1={lambda1}, λ2={lambda2}, λ3={lambda3}")
        
        interpolated_model = InterpolatedNGramModel(train_data, lambda1, lambda2, lambda3)

        # Perplexities on training set
        print("Perplexity on Train Set:", interpolated_model.perplexity(train_data))

        # Perplexities on development set
        print("Perplexity on Dev Set:", interpolated_model.perplexity(dev_data))

    # Evaluate on test set with best lambdas found
    best_lambdas = (0.1, 0.3, 0.6)
    best_interpolated_model = InterpolatedNGramModel(train_data, *best_lambdas)

    print("\nBest Lambda Perplexities on Test Set:")
    print("Perplexity on Test Set:", best_interpolated_model.perplexity(test_data))

    # HDTV Test
    print("\nPerplexity on HDTV Test Set with Best Lambdas:", best_interpolated_model.perplexity(hdtv_test_data))

