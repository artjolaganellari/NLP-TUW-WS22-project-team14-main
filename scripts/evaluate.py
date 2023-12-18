import argparse

from tuwnlpie import logger
from tuwnlpie.milestone1.model import SimpleNBClassifier
from tuwnlpie.milestone1.utils import (
    calculate_tp_fp_fn,
    read_docs_from_csv,
    split_train_dev_test,
)

from tuwnlpie.milestone2.model import BoWClassifier
#from tuwnlpie.milestone2.utils import IMDBDataset, Trainer

from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report, 
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

import torch


def eval_model(model, X_test, y_test, binary=False):
    y_pred = model.predict(X_test)
    
    if binary:
        p, r, f, support = precision_recall_fscore_support(y_test, y_pred, labels=y_test.unique())
        support = f'number of occurences of each label (from {y_test.unique()}): {support}\n'
        print(f'Model eval')
        print(f'precision: {p}\nrecall: {r}\nf-score: {f}\n{support}')
    else:
        avgs = [None, 'micro', 'macro']
        for avg in avgs:
            p, r, f, support = precision_recall_fscore_support(y_test, y_pred, labels=y_test.unique(), average=avg)

            if not avg:
                avg = f'separate for each label: {y_test.unique()}'
            print(f'Model eval ({avg})')

            if support is None:
                support = ''
            else:
                support = f'number of occurences of each label: {support}\n'
            print(f'precision: {p}\nrecall: {r}\nf-score: {f}\n{support}')

    cm = confusion_matrix(y_test, y_pred, labels=y_test.unique(), normalize='true')

    _, ax = plt.subplots(figsize=(5, 3))
    ax.set(xlabel='true label', ylabel='predicted label')
    sns.heatmap(cm, annot=True, xticklabels=y_test.unique(), yticklabels=y_test.unique(), ax=ax)

    plt.show()


def evaluate_milestone1(test_data, saved_model, split=False):
    model = SimpleNBClassifier()
    model.load_model(saved_model)
    docs = read_docs_from_csv(test_data)
    test_docs = None

    if split:
        _, _, test_docs = split_train_dev_test(docs)
    else:
        test_docs = docs

    y_true = []
    y_pred = []
    for doc in test_docs:
        pred = model.predict_label(doc[0])
        y_true.append(doc[1])
        y_pred.append(pred)
        logger.info(f"Predicted: {pred}, True: {doc[1]}")

    tp, fp, fn, precision, recall, fscore = calculate_tp_fp_fn(y_true, y_pred)

    print("Statistics:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {fscore}")

    return


def evaluate_milestone2(test_data, saved_model, split=False):
    logger.info("Loading data...")
    dataset = IMDBDataset(test_data)
    model = BoWClassifier(dataset.OUT_DIM, dataset.VOCAB_SIZE)
    model.load_model(saved_model)
    trainer = Trainer(dataset=dataset, model=model)

    logger.info("Evaluating...")
    test_loss, test_prec, test_rec, test_fscore = trainer.evaluate(
        dataset.test_iterator
    )

    print("Statistics:")
    print(f"Loss: {test_loss}")
    print(f"Precision: {test_prec}")
    print(f"Recall: {test_rec}")
    print(f"F1-Score: {test_fscore}")

    return


def print_eval_metrics(y_true, y_pred, labels):
    logger.info('Classification report:')
    print(classification_report(y_true, y_pred, digits=2))
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)

    ax.set(title='Confusion matrix', xlabel='Predicted', ylabel='True', xticklabels=labels, yticklabels=labels)
    plt.show()

def evaluate_lstm(model, test_loader, class_labels=['neutral', 'treat', 'cause'], device='cpu'):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (sentence, sentence_len), labels in test_loader:           
            labels = labels.to(device)
            sentence = sentence.to(device)
            sentence_len = sentence_len.to(device)

            output = model.predict(sentence, sentence_len)

            y_true += labels.tolist()
            y_pred += output.tolist()

    print_eval_metrics(y_true, y_pred, class_labels)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--test-data", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "-sm", "--saved-model", type=str, required=True, help="Path to saved model"
    )
    parser.add_argument(
        "-sp", "--split", default=False, action="store_true", help="Split data"
    )
    parser.add_argument(
        "-m", "--milestone", type=int, choices=[1, 2], help="Milestone to evaluate"
    )

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    test_data = args.test_data
    model = args.saved_model
    split = args.split
    milestone = args.milestone

    if milestone == 1:
        evaluate_milestone1(test_data, model, split=split)
    elif milestone == 2:
        evaluate_milestone2(test_data, model, split=split)
