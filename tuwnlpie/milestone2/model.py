import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tuwnlpie import logger


class BoWClassifier(nn.Module):  # inheriting from nn.Module!
    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.
        # Torch defines nn.Linear(), which provides the affine map.
        # Note that we could add more Linear Layers here connected to each other
        # Then we would also need to have a HIDDEN_SIZE hyperparameter as an input to our model
        # Then, with activation functions between them (e.g. RELU) we could have a "Deep" model
        # This is just an example for a shallow network
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec, sequence_lens):
        # Ignore sequence_lens for now!
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        # Softmax will provide a probability distribution among the classes
        # We can then use this for our loss function
        return F.log_softmax(self.linear(bow_vec), dim=1)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))


class RNNClassifier(nn.Module):
    def __init__(self,embed_len,hidden_dim,n_layers,target_classes,vocablen):
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocablen, embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, len(target_classes))

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(self.n_layers, len(X_batch), self.hidden_dim))
        return self.linear(output[:,-1])
            
class StackedRNNClassifier(nn.Module): #stacked rnn classifier where hidden dimension size increases by 10 for each stack
    def __init__(self,embed_len,hidden_dim,n_layers,target_classes,vocablen):
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        super(StackedRNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocablen, embedding_dim=embed_len)
        self.rnn1 = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim+10, num_layers=1, batch_first=True)
        self.rnn3 = nn.RNN(input_size=hidden_dim+10, hidden_size=hidden_dim+20, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim+20, len(target_classes))

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn1(embeddings, torch.randn(self.n_layers, len(X_batch), self.hidden_dim))
        output, hidden = self.rnn2(output, torch.randn(self.n_layers, len(X_batch), self.hidden_dim+10))
        output, hidden = self.rnn3(output, torch.randn(self.n_layers, len(X_batch), self.hidden_dim+20))
        return self.linear(output[:,-1])
        
        
class BidirectionalRNNClassifier(nn.Module):
    def __init__(self,embed_len,hidden_dim,n_layers,target_classes,vocablen):
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        super(BidirectionalRNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocablen, embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers,
                          batch_first=True, bidirectional=True) ## Bidirectional RNN
        self.linear = nn.Linear(2*hidden_dim, len(target_classes)) ## Input dimension are 2 times hidden dimensions due to bidirectional results

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(2*self.n_layers, len(X_batch), self.hidden_dim))
        return self.linear(output[:,-1])


#Bi LSTM classifier
class LSTM(nn.Module):

    def __init__(self, vocab_length, lstm_input_size=300, num_layers=1, dimension=64, dropout_rate=0.2, num_classes=3):
        super(LSTM, self).__init__()
        self.dim = dimension

        ## Embedding layer to convert sentences to vectors
        ## Bi-directional LSTM model for feature learning
        ## Dropout layer to prevent overfitting
        ## Linear and Softmax layer to map LSTM features to classes
        self.embedding = nn.Embedding(vocab_length, lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=dimension,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=dropout_rate)

        self.linear = nn.Linear(2*dimension, num_classes)
        self.classifier = nn.Softmax(dim=1)

    def forward(self, x, x_lengths):
        x_embedded = self.embedding(x)

        x_packed = pack_padded_sequence(x_embedded, x_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        x_packed_feats, _ = self.lstm(x_packed)
        x_padded_feats, _ = pad_packed_sequence(x_packed_feats, batch_first=True)

        # combine forward and backward features from lstm
        out_feats_forward = x_padded_feats[range(len(x_padded_feats)), x_lengths - 1, :self.dim]
        out_feats_backward = x_padded_feats[:, 0, self.dim:]
        out_feats_full = torch.cat((out_feats_forward, out_feats_backward), 1)
        out_feats_full = self.drop(out_feats_full)

        x_feats = self.linear(out_feats_full)
        x_feats = torch.squeeze(x_feats, 1)
        soft_prediction = self.classifier(x_feats)

        return soft_prediction

    def predict(self, x, x_lengths):
        # choose class with the highest probability
        softmax_pred = self.forward(x, x_lengths)
        y_pred = torch.argmax(softmax_pred, dim=1)
        return y_pred

    def save(self, out_path):
        if not out_path:
            raise ValueError('Provided out path is none, model can\'t be saved!')
        
        state_dict = {'model_state_dict': self.state_dict()}
        
        torch.save(state_dict, out_path)
        logger.info(f'Saved model to: {out_path}')

    def load(self, load_path, device):
        if not load_path:
            raise ValueError('Provided load path is none, model can\'t be loaded!')
        
        state_dict = torch.load(load_path, map_location=device)
        logger.info(f'Loaded model from: {load_path}')
        
        self.load_state_dict(state_dict['model_state_dict'])

 #Simple LSTM classifier       
class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim = 50, hidden_dim = 50, lstm_input_size=300, num_layers=1, dropout_rate = 0.2) :
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(p=dropout_rate)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim,
                            num_layers,
                            batch_first=True
                            )

        self.linear = nn.Linear(hidden_dim, 3)
        
    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.drop(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out

    def predict(self, x, x_lengths):
        # choose class with the highest probability
        softmax_pred = self.forward(x, x_lengths)
        y_pred = torch.argmax(softmax_pred, dim=1)
        return y_pred

    def save(self, out_path):
        if not out_path:
            raise ValueError('Provided out path is none, model can\'t be saved!')
        
        state_dict = {'model_state_dict': self.state_dict()}
        
        torch.save(state_dict, out_path)
        logger.info(f'Saved model to: {out_path}')

    def load(self, load_path, device):
        if not load_path:
            raise ValueError('Provided load path is none, model can\'t be loaded!')
        
        state_dict = torch.load(load_path, map_location=device)
        logger.info(f'Loaded model from: {load_path}')
        
        self.load_state_dict(state_dict['model_state_dict'])
