import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
plt.switch_backend('agg')

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

# download dataset and split dataset into train and test
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL) 
# this is slow, instead of downloading by this we can use the source code to deal with the dataset

# build a validation set
train_data, valid_data = train_data.split(random_state = random.seed(SEED),split_ratio = 0.8)

print(f'Number of training examples: {len(train_data)}') # 20000
print(f'Number of validation examples: {len(valid_data)}') # 5000
print(f'Number of testing examples: {len(test_data)}') # 25000

# build vocabulary (each word with one vector representation)
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(TEXT.vocab.itos[:10]) # type torchtext.vocab.Vocab
print(LABEL.vocab.stoi) # how to transfer list/numpy to torchtext.vocab.Vocab?

# build iterators, each itaration returns a batch
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
	(train_data, valid_data, test_data),
	batch_size = BATCH_SIZE, device = device)
# seq_len * batch_size
# batch = next(iter(valid_iterator))
# batch.label
# batch.text.shape

# word averaging embedding
class WordAVGModel(nn.Module):
	def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
		super(WordAVGModel, self).__init__()
		self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
		self.linear = nn.Linear(embedding_size, output_size)

	def forward(self, text):
		embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
		embedded = embedded.permute(1,0,2) # [batch_size, seq_len, embedding_size]
		pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze() # [batch_size, embedding_size]
		return self.linear(pooled)

VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = WordAVGModel(vocab_size = VOCAB_SIZE,
	embedding_size = EMBEDDING_SIZE,output_size = OUTPUT_SIZE,pad_idx=PAD_IDX)

# model parameters
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

#count_parameters(model)

# use glove to initiate model
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_INX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_INX] = torch.zeros(EMBEDDING_SIZE)

# define optimizer and criteria to train the model
optimizer = torch.optim.Adam(model.parameters()); model = model.to(device)
crit = nn.BCEWithLogitsLoss(); crit = crit.to(device)

# calculate acc
def binary_accuracy(preds, y):
	rounded_preds = torch.round(torch.sigmoid(preds))
	correct = (rounded_preds == y).float()
	acc = correct.sum()/len(correct)
	return acc

# train the model (define the training process)
def train(model, iterator, optimizer, crit):
	epoch_loss, epoch_acc = 0., 0.
	model.train()
	total_len = 0.
	for batch in iterator:
		batch.text = torch.tensor(batch.text,dtype=torch.float)
		preds = model(batch.text).squeeze() #[batch_size]
		loss = crit(preds, batch.label)
		acc = binary_accuracy(preds, batch.label)

		# sgd
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()

		epoch_loss += loss.item() * len(batch.label)
		epoch_acc += acc.item() * len(batch.label)
		total_len += len(batch.label)

	return epoch_loss/total_len, epoch_acc/total_len

# evaluate the model (define the evaluation process)
def evaluate(model, iterator, crit):
	epoch_loss, epoch_acc = 0., 0.
	model.eval()
	total_len = 0.
	for batch in iterator:
		preds = model(batch.text).squeeze() #[batch_size]
		loss = crit(preds, batch.label)
		acc = binary_accuracy(preds, batch.label)

		epoch_loss += loss.item() * len(batch.label)
		epoch_acc += acc.item() * len(batch.label)
		total_len += len(batch.label)

	return epoch_loss/total_len, epoch_acc/total_len
def PlotAUC(model, iterator, file_prefix):
    # label(nuc) = 0, label(cyto) = 1
    test_loss, test_acc = 0., 0.
    total_len = 0.
    pre_label, pre_label_score, true_label = [],[],[]
    with torch.no_grad():
	    for batch in iterator:
	    	preds = model(batch.text).squeeze()
	    	loss =  crit(preds, batch.label)
	    	acc = binary_accuracy(preds, batch.label)

	    	test_loss += loss.item() * len(batch.label)
	    	test_acc += acc.item() * len(batch.label)
	    	total_len += len(batch.label)
	    	# get the pre_label and true_label for all predict dataset
	    	pre_label_score += torch.sigmoid(preds).detach().cpu().tolist()
	    	pre_label += torch.round(torch.sigmoid(preds)).detach().cpu().tolist()
	    	true_label += batch.label.detach().cpu().tolist()
    
    loss = test_loss / total_len
    accuracy = test_acc / total_len
    #loss,accuracy = DeepLoc.evaluate(X_test,y_test,verbose=2)
    #y_pred_proba = DeepLoc.predict_proba(X_test)
    #y_pred = DeepLoc.predict(X_test)
    #pre_label = list(torch.round(preds))
    #pre_label = list(np.argmax(y_pred,axis=1))
    #true_label = list(y_test)

    TP=0.
    FP=0.
    FN=0.
    TN=0.
    num = len(pre_label)
    for i in range(num):
        if np.logical_and(pre_label[i]==1,pre_label[i]==true_label[i]):
            TP += 1
        elif np.logical_and(pre_label[i]==0,pre_label[i]==true_label[i]):
            TN += 1
        elif np.logical_and(pre_label[i]==1,pre_label[i]!=true_label[i]):
            FP += 1
        else:
            FN += 1

    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    MCC = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
    F1 = 2*((TP/(TP+FP)*TP/(FN+TP))/(TP/(TP+FP)+TP/(FN+TP)))

    evaluate = open(file_prefix+'evaluate.txt','w')
    evaluate.writelines("Loss:%3f\tAccuracy:%3f\tSensitivity:%.3f\tSpecificity:%3f\tMCC:%3f\tF1:%3f" %(loss,accuracy,Sensitivity,Specificity,MCC,F1))  
    evaluate.close()
    
    fpr, tpr, _ = metrics.roc_curve(true_label, pre_label_score)
    # tpr = sensitivity = TP/ (TP + FN) 
    # fpr = FP / (FP+TN)

    roc_auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(roc_auc))
    plt.legend(loc='best')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(file_prefix + '_AUC.pdf', format='pdf')
    return loss,accuracy,Sensitivity,Specificity,MCC,F1

N_EPOCHS = 15
best_valid_acc = 0.
for epoch in range(N_EPOCHS):
		train_loss, train_acc = train(model, train_iterator, optimizer, crit)
		valid_loss, valid_acc = evaluate(model, valid_iterator, crit)

		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc
			torch.save(model.state_dict(), "wordAVG-model.pth")

		print("Epoch", epoch, "Train Loss", train_loss, "Train Acc", train_acc)
		print("Epoch", epoch, "Valid Loss", valid_loss, "Valid Acc", valid_acc)

# now let's do some prediction!
model.load_state_dict(torch.load("wordAVG-model.pth"))
import spacy
nlp = spacy.load("en")

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device) # seq_len long tensor is important
    tensor = tensor.unsqueeze(1) # seq_len * batch_size(1)
    pred = torch.sigmoid(model(tensor))
    return pred.item()

sentence = u"This film is horrible!"
predict_sentiment(u"This film is horrible!")
predict_sentiment(u"This film is terrific!")


##### use RNN lstm model instead #####
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, hidden_size, dropout):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=2)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
        embedded  = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded)
        
        # hidden: 2 * batch_size * hidden_size
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        hidden = self.dropout(hidden.squeeze())
        return self.linear(hidden)

model = RNNModel(vocab_size=VOCAB_SIZE, 
                 embedding_size=EMBEDDING_SIZE, 
                 output_size=OUTPUT_SIZE, 
                 pad_idx=PAD_IDX, 
                 hidden_size=100, 
                 dropout=0.3)
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()

model = model.to(device)
crit = crit.to(device)

N_EPOCHS = 10
best_valid_acc = 0.
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)
    
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), "lstm-model.pth")
        
    print("Epoch", epoch, "Train Loss", train_loss, "Train Acc", train_acc)
    print("Epoch", epoch, "Valid Loss", valid_loss, "Valid Acc", valid_acc)


###### use RNN GRU model ######
class RNNModel_GRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, hidden_size, dropout):
        super(RNNModel_GRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, num_layers=1)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
        embedded  = self.dropout(embedded)
        output, hidden = self.gru(embedded)
        
        # hidden: 2 * batch_size * hidden_size
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        hidden = self.dropout(hidden.squeeze())
        return self.linear(hidden)

# define the hyper-parameters
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNNModel_GRU(vocab_size=VOCAB_SIZE, 
                 embedding_size=EMBEDDING_SIZE, 
                 output_size=OUTPUT_SIZE, 
                 pad_idx=PAD_IDX, 
                 hidden_size=100, 
                 dropout=0.3)

pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()

model = model.to(device)
crit = crit.to(device)


# put the embedding layer out of the model
class RNNModel_GRU2(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, hidden_size, dropout):
        super(RNNModel_GRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, num_layers=1)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
        embedded  = self.dropout(embedded)
        output, hidden = self.gru(embedded)
        
        # hidden: 2 * batch_size * hidden_size
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        hidden = self.dropout(hidden.squeeze())
        return self.linear(hidden)

# define the hyper-parameters
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNNModel_GRU(vocab_size=VOCAB_SIZE, 
                 embedding_size=EMBEDDING_SIZE, 
                 output_size=OUTPUT_SIZE, 
                 pad_idx=PAD_IDX, 
                 hidden_size=100, 
                 dropout=0.3)


pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, padding_idx=PAD_IDX)
embedded = embed(text)

optimizer = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()

model = model.to(device)
crit = crit.to(device)


N_EPOCHS = 10
best_valid_acc = 0.
for epoch in range(N_EPOCHS):
	train_loss, train_acc = train(model, train_iterator, optimizer, crit)
	valid_loss, valid_acc = evaluate(model, valid_iterator, crit)
    
    # if valid_acc > best_valid_acc:
    #     best_valid_acc = valid_acc
    #     torch.save(model.state_dict(), "gru-model.pth")

	if valid_acc > best_valid_acc:
		print("The validation accuracy has improved from", best_valid_acc,"to", valid_acc)
		best_valid_acc = valid_acc
		print("Saving the best model with state...")
		state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), }
		torch.save(state, "IMDB_GRU-model-state-retainGraph.pth.tar")
        #torch.save(model.state_dict(), "GRU-model.pth")
        
	print("Epoch", epoch, "Train Loss", train_loss, "Train Acc", train_acc)
	print("Epoch", epoch, "Valid Loss", valid_loss, "Valid Acc", valid_acc)

# method to load the saved model
def load_checkpoint(model, optimizer, filename='IMDB_GRU-model-state.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

# load the model with state
model, optimizer, start_epoch = load_checkpoint(model, optimizer)
model = model.to(device)
# now individually transfer the optimizer parts...
for state in optimizer.state.values():
	for k, v in state.items():
		if isinstance(v, torch.Tensor):
			state[k] = v.to(device)

loss,accuracy,Sensitivity,Specificity,MCC,F1 = PlotAUC(model, test_iterator, "IMDB_GRU-model-state-retainGraph")

import spacy
nlp = spacy.load("en")

# example 
predict_sentiment(u"This film is horrible!")
predict_sentiment(u"This film is terrific!")

# use backward step get the autograd derivative
def pre_process(sentence):
	tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
	indexed = [TEXT.vocab.stoi[t] for t in tokenized]
	tensor = torch.tensor(indexed,dtype=torch.float).to(device) # Use float type to allow gradient
	tensor = tensor.unsqueeze(1)
	tensor.requires_grad_()
	return tensor # return a tensor with gradient avaliable


input_1 = u"This film is horrible!"
preprocess_1 = pre_process(input_1) # requires_grad = True

model.eval()

scores = model(preprocess_1)

input_1,output_1 = get_prediction(u"We thought this movie is wonderful")
input_1 = input_1.float()
input_1.requires_grad=True
y = torch.tensor([0.]).to(device)
y = torch.tensor([1.],requires_grad=True).to(device)

torch.autograd.grad(output_1, input_1, retain_graph=True,allow_unused=True) #(None,)
torch.autograd.grad(output_1, y, retain_graph=True,allow_unused=True) #(none,)

model.eval()
loss = crit(torch.sigmoid(output_1),y).to(device)
df_do = loss.backward()

torch.autograd.grad(output_1, input_1)


df_do = torch.autograd.backward(loss, )
inputGrads = model:backward(X, df_do)

df_do = crit:backward(output, y)
inputGrads = model:backward(X, df_do)
inputGrads = torch.abs(inputGrads)
inputGrads = torch.cmul(inputGrads,X)
-- inputGrads = inputGrads:max(2)
inputGrads = inputGrads:view(opt.seq_length,4)
score = output[1]:exp()[1]

