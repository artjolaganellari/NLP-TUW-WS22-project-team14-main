import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from tuwnlpie.milestone1.utils import (
    split_train_dev_test,
    prepare_dataset,
    lemmatize_dataset,
    transform_to_dictionary,
    train_NaiveBayes,
    show_features,
    evaluate_nltk_nb,
    test_prediction_for_new_sentence,
    split_train_test_sklearn,
    train_multi_nb_classifier,
    train_svm_classifier,
    undersampling_dataset
)

from tuwnlpie.milestone2.model import RNNClassifier, BidirectionalRNNClassifier, LSTM
from tuwnlpie.milestone2.utils import (
    TrainModel,
    EvaluatePerformance,
    get_loaders_lstm, 
    prepare_dataset_lstm
)

from scripts.evaluate import eval_model, evaluate_lstm
from scripts.train import train_lstm
from tuwnlpie import logger

import torch
from torch.optim import Adam

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--milestone', default='ms1', help='Available options are: ms1, ms2, final\nDefault: ms1')           
parser.add_argument('--model', default='nb-nltk', 
    help='Available options are:\n' +
    'ms1: nb-nltk for naive bias with nltk, nb-scikit for multi naive bias with scikit, svm for support vector machine classifier\n' +
    'ms2: cnn for simple CNN, rnn for RNN, bi-rnn for bi-directional RNN, bi-lstm for bi-directional LSTM\n'+
    'final: \n' +
    'Default: nb-nltk')
parser.add_argument('--seed', default=1234, help="Random seed. Default: 1234")


def run_ms1(model_name, seed):
    logger.info(f"Starting MS1 with {model_name}...")

    logger.info("Reading data...")
    data = prepare_dataset("./data/food_disease_dataset.csv")

    logger.info("Transforming data... This might take a while")
    transformed_data = transform_to_dictionary(data) if model_name  == 'nb-nltk' else lemmatize_dataset(data)

    if model_name == 'nb-nltk':
        logger.info("Data transformation is finished! Starting model training...")
        train_data, dev_data, test_data = split_train_dev_test(transformed_data)
        model = train_NaiveBayes(train_data)

        logger.info("Model training is done! Starting model evaluation...")
        evaluate_nltk_nb(model, test_data)
        return

    model_dict = {'nb-scikit': train_multi_nb_classifier, 'svm': train_svm_classifier}
    X_train, X_test, y_train, y_test = split_train_test_sklearn(transformed_data, random_state=seed)
    logger.info("Data transformation is finished! Starting model training...")

    model = model_dict[model_name](X_train, y_train)
    logger.info("Model training is done! Starting model evaluation...")

    eval_model(model, X_test, y_test)


def run_ms2(model_name, seed):
    logger.info(f"Starting MS2 with {model_name}...")

    logger.info("Reading data...")
    torch.manual_seed(seed)
    prepare_dataset_lstm()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size=32
    train_loader, valid_loader, test_loader, vocab_length = get_loaders_lstm(batch_size=batch_size, device=device)

    logger.info("Transforming data... This might take a while")
    if model_name == 'cnn':
        return

    if 'rnn' in model_name: 
        embed_len = 50
        hidden_dim = 50
        n_layers= 1
        target_classes = ["cause", "treat", "neutral"]
        epochs = 15
        learning_rate = 0.01

        if model_name == 'rnn':
            rnn_classifier = RNNClassifier(embed_len, hidden_dim, n_layers, target_classes, vocab_length)
        elif model_name =='bi-rnn':
            rnn_classifier = BidirectionalRNNClassifier(embed_len, hidden_dim, n_layers, target_classes, vocab_length)
        
        optimizer = Adam(rnn_classifier.parameters(), lr=learning_rate)
        TrainModel(rnn_classifier, optimizer, train_loader, test_loader, epochs)
        EvaluatePerformance(rnn_classifier, test_loader, target_classes)

        return

    if model_name == 'bi-lstm':
        model = LSTM(vocab_length=vocab_length, dropout_rate=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_lstm(model=model, optimizer=optimizer, num_epochs=20, train_loader=train_loader, valid_loader=valid_loader, device=device)

        best_model = LSTM(588).to(device)

        best_model.load('./data/bi_lstm.pt', device)
        evaluate_lstm(best_model, test_loader, device=device)


def run_final(model_name, seed):
    return


def check_args(args):
    ms_model_dict = {
        'ms1': ['nb-nltk', 'nb-scikit', 'svm'],
        'ms2': ['cnn', 'rnn', 'bi-rnn', 'bi-lstm'],
        'final': []
    }

    if args.milestone not in ms_model_dict.keys():
        raise ValueError(f"Unvalid --milestone value: {args.milestone}. Available values are: {list(ms_model_dict.keys())}")

    if args.model not in ms_model_dict[args.milestone]:
        raise ValueError(f"Unvalid --model value: {args.model}. " + 
        f"Available values for {args.milestone} milestone are: {ms_model_dict[args.milestone]}")


def main():
    args = parser.parse_args()
    check_args(args)

    milestone_dict = {'ms1': run_ms1, 'ms2': run_ms2, 'final': run_final}
    milestone_dict[args.milestone](args.model, args.seed)
    

if __name__ == "__main__":
    main()