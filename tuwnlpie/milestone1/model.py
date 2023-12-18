import csv
from collections import Counter, defaultdict
from math import exp, log

from tuwnlpie import logger

class SimpleNBClassifier:
    def __init__(self):
        self.word_count = Counter()
        self.count_by_class = defaultdict(Counter)
        self.labels = set()

    def save_model(self, filename):
        with open(filename, "w") as f:
            for word, weights in self.weights.items():
                label_weights = "\t".join(
                    f"{label}:{weight}" for label, weight in weights.items()
                )
                f.write(f"{word}\t{label_weights}\n")

    def load_model(self, filename):
        self.weights = {}
        with open(filename) as f:
            for line in f:
                line_split = line.strip().split("\t")
                word = line_split[0]
                label_weights = line_split[1:]
                self.weights[word] = {
                    label: float(weight)
                    for label, weight in (x.split(":") for x in label_weights)
                }
        self.labels = set(self.weights[list(self.weights.keys())[0]].keys())

    def count_words(self, docs):
        logger.info("Counting words")
        for words, label in docs:
            self.labels.add(label)
            for word in set(words):
                self.word_count[word] += 1
                self.count_by_class[label][word] += 1

    def calculate_weights(self):
        logger.info("Calculating weights")
        self.weights = {
            word: {
                label: log(
                    (self.count_by_class[label][word] + 1)
                    / (count + len(self.word_count))
                )
                for label in self.labels
            }
            for word, count in self.word_count.items()
        }

        logger.info("Finished training")

    def get_doc_weights(self, doc):
        return {
            label: sum(
                self.weights[word][label] if word in self.weights else log(1)
                for word in doc
            )
            for label in self.labels
        }

    def predict_label(self, doc):
        doc_weights = self.get_doc_weights(doc)
        return sorted(doc_weights.items(), key=lambda x: -x[1])[0][0]
