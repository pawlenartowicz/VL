import torch
import wandb
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# training loop for binary classification with one dimensional output
def training_loop(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, epochs, device, savedir):
    for epoch in range(epochs):
        total_loss_train = 0
        best_acc = 0
        # to device
        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device).float()
            # train input are vectors
            train_input = train_input.to(device).float()
            output = model(train_input)
            output = output.squeeze(1)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()


            del output, train_label, train_input

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        total_acc_val = 0
        total_loss_val = 0
        model.eval()
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device).float()
                val_input = val_input.to(device).float()

                output = model(val_input)
                output = output.squeeze(1)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                # accuracy
                acc = torch.sum(torch.round(torch.sigmoid(output)) == val_label).item() / len(val_label)

                total_acc_val += acc

                del acc, output, val_label, val_input

        if total_acc_val / len(val_dataloader) > best_acc:
            best_acc = total_acc_val / len(val_dataloader)
            torch.save(model.state_dict(), savedir)

        wandb.log({"Train Loss": total_loss_train / len(train_dataloader),
                   "Validation Loss": total_loss_val / len(val_dataloader),
                   "Validation Accuracy": total_acc_val / len(val_dataloader),
                   "learning rate": optimizer.param_groups[0]['lr']})

        # print results
        print(f"Epoch {epoch + 1} of {epochs} / Train Loss: {total_loss_train / len(train_dataloader)} "
              f"/ Validation Loss: {total_loss_val / len(val_dataloader)} / Validation Accuracy: "
              f"{total_acc_val / len(val_dataloader)}")
