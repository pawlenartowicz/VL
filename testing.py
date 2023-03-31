import sys, os
import torch
sys.path.insert(0, r'D:\GitHub\ergodicity_1991\simple_net')
from dataset_and_model import Dataset, Simple_Net
from preparing_data import load_test

df_test, label_dict = load_test()

number_of_responses = 100
test = Dataset(df_test, number_of_responses)
layer_list = [800, 800, 800, 800]
input_size = number_of_responses * 40

model = Simple_Net(0.3, input_size, layer_list, len(label_dict))
model.load_state_dict(torch.load(r'D:\data\data_hackaton\models\test_8'))

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
    _, pred = torch.max(output, dim=1)
    predictions.extend(pred.tolist())
    reals.extend(test_label.tolist())

import pickle
with open(r'results\probs.pkl', 'wb') as f:
    pickle.dump(probs, f)

with open(r'results\predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)

with open(r'results\reals.pkl', 'wb') as f:
    pickle.dump(reals, f)




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




