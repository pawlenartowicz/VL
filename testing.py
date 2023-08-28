import sys, os
import torch
import pandas as pd
from dataset_and_model import Dataset, Simple_Net

df_train = pd.read_csv('data/df_train_50_0_01.csv')
df_val = pd.read_csv('data/df_val_50_0_01.csv')
df_test = pd.read_csv('data/df_test_50_0_01.csv')
number_of_responses = 500

test = Dataset(df_test, number_of_responses)
layer_list = [1000, 500, 100]
input_size = number_of_responses * 40

model = Simple_Net(0.3, input_size, layer_list, 1)
savedir = r'D:\GitHub\VL\models\test_1.pt'

model.load_state_dict(torch.load(savedir))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

batch_size = 10000
test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

predictions = []
reals = []
model.eval()
probs = []
for test_input, test_label in test_dataloader:
    test_label = test_label.to(device).long()
    # train input are vectors
    test_input = test_input.to(device).float()
    output = model(test_input)
    probs.extend(output.tolist())
    predictions.extend([1 if output[i] > 0.5 else 0 for i in range(len(output))])
    reals.extend(test_label.tolist())

# import pickle
# with open(r'results\probs.pkl', 'wb') as f:
#     pickle.dump(probs, f)
#
# with open(r'results\predictions.pkl', 'wb') as f:
#     pickle.dump(predictions, f)
#
# with open(r'results\reals.pkl', 'wb') as f:
#     pickle.dump(reals, f)




acc = sum([1 if predictions[i] == reals[i] else 0 for i in range(len(predictions))]) / len(predictions)
# reverse
label_dict = {v: k for k, v in label_dict.items()}
prediction_name = [label_dict[pred] for pred in predictions]
real_name = [label_dict[real] for real in reals]
# confusion matrix
from sklearn.metrics import confusion_matrix

c = confusion_matrix(reals, predictions)
# plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(c, annot=True, fmt='d')




