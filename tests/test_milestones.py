import os
from tuwnlpie.milestone1.utils import read_docs_from_csv, split_train_dev_test
from tuwnlpie.milestone1.model import SimpleNBClassifier

from tuwnlpie.milestone2.utils import IMDBDataset

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data.csv")


def test_reader():
    docs = read_docs_from_csv(SAMPLE_PATH)
    labels = [doc[1] for doc in docs]
    positive = labels.count("positive")
    negative = labels.count("negative")

    assert len(docs) == 8
    assert positive == 4
    assert negative == 4


def test_split():
    docs = read_docs_from_csv(SAMPLE_PATH)
    train_docs, dev_docs, test_docs = split_train_dev_test(docs)

    assert len(train_docs) == 6


def test_naive_bayes():
    docs = read_docs_from_csv(SAMPLE_PATH)
    train_docs, dev_docs, test_docs = split_train_dev_test(docs)

    model = SimpleNBClassifier()
    model.count_words(train_docs)
    model.calculate_weights()

    doc1 = ("This", "movie", "is", "nice")

    assert model.predict_label(doc1) == "positive"


def test_dataloader():
    dataset = IMDBDataset(SAMPLE_PATH)

    assert len(dataset.tr_data_loader) == 4
    assert dataset.VOCAB_SIZE == 267
    assert dataset.OUT_DIM == 2


if __name__ == "__main__":
    test_dataloader()
