import sys, os
import torch
from dataset_and_model import Dataset, Simple_Net, CNN_Discriminator
from training_loop import training_loop
import wandb
from creating_training_datasets_2 import MixedGenerator as mixed_generator
from transformers import get_linear_schedule_with_warmup
from attention_model import ClassifierWithAttention
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

overwrite = False

if not os.path.exists('data/df_train_50_0_01.csv') or overwrite == True:
    list_of_tuples = mixed_generator(numberOfJunkQuestionaries = 1000, junkRatio=(0.4,0.5), maxDispersion=0.1, bootstrap=True)
    data, labels = zip(*list_of_tuples)

    # concat data
    data = np.concatenate(data, axis=0)


    number_of_responses = int(len(data) / len(labels))
    # data to dataframe
    df = pd.DataFrame(data)
    labels_list = [[label] * number_of_responses for label in labels]
    labels_list = [item for sublist in labels_list for item in sublist]
    df['label'] = labels_list
    questionnaire_number = [i for i in range(1, len(labels) + 1)]
    questionnaire_number_list = [ [i] * number_of_responses for i in questionnaire_number]
    questionnaire_number_list = [item for sublist in questionnaire_number_list for item in sublist]
    df['questionnaire_number'] = questionnaire_number_list

    # split the questionnaire_numbers into train, val, test using random
    questionnaire_numbers = df['questionnaire_number'].unique()
    df_train = np.random.choice(questionnaire_numbers, int(len(questionnaire_numbers) * 0.8), replace=False)
    df_val = np.random.choice([i for i in questionnaire_numbers if i not in df_train], int(len(questionnaire_numbers) * 0.1), replace=False)
    df_test = [i for i in questionnaire_numbers if i not in df_train and i not in df_val]

    df_train = df[df['questionnaire_number'].isin(df_train)]
    df_val = df[df['questionnaire_number'].isin(df_val)]
    df_test = df[df['questionnaire_number'].isin(df_test)]
    # reset index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train.to_csv('data/df_train.csv', index=False)
    df_val.to_csv('data/df_val.csv', index=False)
    df_test.to_csv('data/df_test.csv', index=False)

else:
    df_train = pd.read_csv('data/df_train_50_0_01.csv')
    df_val = pd.read_csv('data/df_val_50_0_01.csv')
    df_test = pd.read_csv('data/df_test_50_0_01.csv')
    number_of_responses = 500

train, val= Dataset(df_train, number_of_responses),\
                   Dataset(df_val, number_of_responses)


epochs = 1000
# layer_list = [1000, 500, 100]
input_size = number_of_responses * 40
savedir = r'D:\GitHub\VL\models\test_1.pt'
batch_size = 1000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

layer_list = [100, 100, 100, 100]
model = Simple_Net(0.3, input_size, layer_list, 1)
# model = ClassifierWithAttention()
# load

criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=5e-4,
                  eps=1e-8,  # Epsilon
                  weight_decay=0.5,
                  amsgrad=True,
                  betas = (0.9, 0.999))

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=500,
                                            num_training_steps= len(train_dataloader) * epochs)
# model to device
model = model.to(device)

wandb.init(project="simple_net", entity="hubertp")
wandb.watch(model, log_freq=5)

training_loop(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, epochs, device, savedir)
