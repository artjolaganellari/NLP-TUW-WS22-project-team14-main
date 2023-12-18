import argparse

import torch
from torch import nn

from tuwnlpie import logger
from tuwnlpie.milestone1.model import SimpleNBClassifier
from tuwnlpie.milestone1.utils import (
    calculate_tp_fp_fn,
    read_docs_from_csv,
    split_train_dev_test,
)

from tuwnlpie.milestone2.utils import (
    save_metrics
)

from tuwnlpie.milestone2.model import BoWClassifier
# from tuwnlpie.milestone2.utils import IMDBDataset, Trainer


def train_milestone1(train_data, save=False, save_path=None):
    model = SimpleNBClassifier()
    docs = read_docs_from_csv(train_data)

    train_docs, dev_docs, test_docs = split_train_dev_test(docs)

    model.count_words(train_docs)
    model.calculate_weights()

    if save:
        model.save_model(save_path)
        logger.info(f"Saved model to {save_path}")

    return


def train_milestone2(train_data, save=False, save_path=None):
    logger.info("Loading data...")
    dataset = IMDBDataset(train_data)
    model = BoWClassifier(dataset.OUT_DIM, dataset.VOCAB_SIZE)
    trainer = Trainer(dataset=dataset, model=model)

    logger.info("Training...")
    trainer.training_loop(dataset.train_iterator, dataset.valid_iterator)

    if save:
        model.save_model(save_path)

    return


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--train-data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save model"
    )
    parser.add_argument(
        "-sp", "--save-path", default=None, type=str, help="Path to save model"
    )
    parser.add_argument(
        "-m", "--milestone", type=int, choices=[1, 2], help="Milestone to train"
    )

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    train_data = args.train_data
    model_save = args.save
    model_save_path = args.save_path
    milestone = args.milestone

    if milestone == 1:
        train_milestone1(train_data, save=model_save, save_path=model_save_path)
    elif milestone == 2:
        train_milestone2(train_data, save=model_save, save_path=model_save_path)


def validate_lstm(model, 
          valid_loader, 
          epoch, 
          num_epochs,
          num_steps,
          i, 
          train_loss, 
          valid_loss, 
          steps, 
          running_loss,
          best_valid_loss,
          criterion = nn.CrossEntropyLoss(),
          out_path = './data',
          device='cpu'):
    model.eval()

    valid_running_loss = 0.0

    with torch.no_grad():
        for (sentence, sentence_len), labels in valid_loader:
            labels = labels.to(device)
            sentence = sentence.to(device)
            sentence_len = sentence_len.to(device)
            output = model(sentence, sentence_len)

            loss = criterion(output, labels)
            valid_running_loss += loss.item()

        average_train_loss = running_loss / 5
        average_valid_loss = valid_running_loss / len(valid_loader)

    train_loss.append(average_train_loss)
    valid_loss.append(average_valid_loss)
    steps.append(i)

    model.train()

    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{num_steps}], Train Loss: {average_train_loss:.3f}, Valid Loss: {average_valid_loss:.3f}')
    
    if average_valid_loss < best_valid_loss:
        best_valid_loss = average_valid_loss
        model.save(f'{out_path}/bi_lstm.pt')
        save_metrics(f'{out_path}/bi_lstm_metrics.pt', train_loss, valid_loss, steps)
    
    return best_valid_loss


def train_lstm(model,
          optimizer,
          train_loader,
          valid_loader,
          criterion = nn.CrossEntropyLoss(),
          num_epochs = 10,
          out_path = './data',
          best_valid_loss = float("Inf"),
          device='cpu'):
    
    loss_acc = 0.0
    i = 0
    train_loss = []
    valid_loss = []
    steps = []

    model.train()
    for epoch in range(num_epochs):
        for (sentence, sentence_len), labels in train_loader:
            labels = labels.to(device)
            sentence = sentence.to(device)
            sentence_len = sentence_len.to(device)
            
            output = model(sentence, sentence_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_acc += loss.item()
            i += 1

            if i % 5 == 0:
                best_valid_loss = validate_lstm(model, 
                                                valid_loader, 
                                                epoch, 
                                                num_epochs,
                                                i*len(train_loader),
                                                i, 
                                                train_loss, 
                                                valid_loss, 
                                                steps, 
                                                loss_acc,
                                                best_valid_loss)
                loss_acc = 0.0
    
    logger.info('Training finished')
    save_metrics(f'{out_path}/bi_lstm_metrics.pt', train_loss, valid_loss, steps)
