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
# build model
# put the embedding layer out of the model
class RNNModel_GRU2(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, hidden_size, dropout):
        super(RNNModel_GRU2, self).__init__()
        #self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, num_layers=1)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embedded_text):
        # embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
        embedded  = self.dropout(embedded_text)
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

model = RNNModel_GRU2(vocab_size=VOCAB_SIZE, 
                 embedding_size=EMBEDDING_SIZE, 
                 output_size=OUTPUT_SIZE, 
                 pad_idx=PAD_IDX, 
                 hidden_size=100, 
                 dropout=0.3)

# The structure of model
# RNNModel_GRU2(
#   (gru): GRU(100, 100, bidirectional=True)
#   (linear): Linear(in_features=200, out_features=1, bias=True)
#   (dropout): Dropout(p=0.3)
# )

# build word vectors
pretrained_embedding = TEXT.vocab.vectors  # [25002, 100]

# initiate pad an unk vectors
TEXT.vocab.vectors[0] = TEXT.vocab.vectors[1] = torch.zeros(EMBEDDING_SIZE)

# embed out of model
embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, padding_idx=PAD_IDX).to(device)
embed.weight.data.copy_(pretrained_embedding)
# initiate the embedding word vectors
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)
# embedded = torch.tensor(embed(text),requires_grad=False)

# set optimizer and criteria
optimizer = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()

model = model.to(device)
crit = crit.to(device)

# model parameters
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
		embedded = torch.tensor(embed(batch.text),requires_grad=False)
		preds = model(embedded).squeeze() #[batch_size]
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
		embedded = torch.tensor(embed(batch.text),requires_grad=False) # [52, 64, 100]
		preds = model(embedded).squeeze() #[batch_size]
		loss = crit(preds, batch.label)
		acc = binary_accuracy(preds, batch.label)

		epoch_loss += loss.item() * len(batch.label)
		epoch_acc += acc.item() * len(batch.label)
		total_len += len(batch.label)

	return epoch_loss/total_len, epoch_acc/total_len

# for model performance and plotting
def PlotAUC(model, iterator, file_prefix):
    # label(nuc) = 0, label(cyto) = 1
    test_loss, test_acc = 0., 0.
    total_len = 0.
    pre_label, pre_label_score, true_label = [],[],[]
    with torch.no_grad():
	    for batch in iterator:
	    	embedded = torch.tensor(embed(batch.text))
	    	preds = model(embedded).squeeze()
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

# start training and evaluation
N_EPOCHS = 10
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
    embedded = torch.tensor(embed(tensor),requires_grad=False)
    pred = torch.sigmoid(model(embedded))
    return pred.item()

predict_sentiment(u"This film is horrible!")
predict_sentiment(u"This film is terrific!")

# for model performance
loss,accuracy,Sensitivity,Specificity,MCC,F1 = PlotAUC(model, test_iterator, "IMDB_GRU_EMDout-model-state-retainGraph")

# tring Saliency now!
# for pre-processing the sentence (sentence to vectors)
def pre_process(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device) # seq_len long tensor is important
    tensor = tensor.unsqueeze(1) # seq_len * batch_size(1)
    embedded = torch.tensor(embed(tensor),requires_grad=True) # here we need to get gradient for saliency computation, use requires_grad=True
    # pred = torch.sigmoid(model(embedded))
    return embedded
# deal with input sentence
input_1 = u"This film is horrible!"
input_1 = u"This movie was sadly under-promoted but proved to be truly exceptional."
input_1 = u"The movie is fantastic I really like it."
preprocess_1 = pre_process(input_1) # requires_grad = True
# we would run the model in evaluation mode
model.train() # if I set model.eval(), an error occur: RuntimeError: cudnn RNN backward can only be called in training mode
model.dropout.eval() # only freeze the drop out layer and batch normalization layer that will generate randomness to the model.
#with torch.no_grad():
'''forward pass through the model to get the scores, note that RNNModel_GRU2 model doesn't perform sigmoid at the end
and we also don't need sigmoid, we need scores, so that's perfect for us.
'''
scores = model(preprocess_1)
'''
backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
score_max with respect to nodes in the computation graph
'''
scores.backward()

'''
Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
across all colour channels.
'''
# saliency, _ = torch.max(preprocess_1.grad.data.abs(),dim=2) # Use the max from all dimension
saliency = preprocess_1.grad.data.abs().squeeze() # we can use the whole dimension of Saliency to present [len_sentence, 100]
saliency_list = saliency.detach().cpu().numpy() # for Saliency plot numpy.ndarray [len_sentence, 100]

 # [This, film, is, horrible, !]
torch.sigmoid(scores) # [0.0065], shows the classification of prediction

# plotting the Saliency heatmap
#plt.figure(figsize=(8, 5))
# generate words list from the whole sentence
words = [tok for tok in nlp.tokenizer(input_1)]
# create the figure
fig = plt.figure(figsize=(10, 5))
# set the sub-figure
ax = fig.add_subplot(111)
ax.set_aspect(aspect=2)
im = plt.imshow(saliency_list, aspect='auto',interpolation='nearest', cmap=plt.cm.Blues)
# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Saliency", rotation=-90, va="bottom")
# set ticks for axis
ax.set_yticks(np.arange(len(words)))
ax.set_xticks(np.arange(len(saliency_list[0]),step = 20))
ax.set_yticklabels(words)
ax.set_title("Saliency heatmap for the emotion prediction")
#ax.colorbar()
plt.savefig( 'SaliencyHeatmap_'+input_1+'.pdf', format='pdf')



