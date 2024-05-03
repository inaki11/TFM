from tqdm import tqdm
import torch
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(model, train_dataloader, optimizer, scheduler, NUM_EPOCHS, tem, lam, loss_fn='cross_entropy'):
    train_losses = []
    for epoch in range(NUM_EPOCHS):
        # Set your model to training mode
        total_loss = 0
        total_train_samples = 0
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                input_ids, attention_mask, labels, _ = data
                # Move batch to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                embbedings, outputs = model(input_ids, attention_mask)
                logits = outputs.squeeze(-1)

                # Compute loss
                if loss_fn == 'cross_entropy':
                    criterion = torch.nn.BCELoss() # torch.nn.BCEWithLogitsLoss()
                    loss = criterion(logits, labels.float())
                elif loss_fn == 'supervised_contrastive':
                    #print("supervised_contrastive")
                    criterion = torch.nn.BCELoss() # torch.nn.BCEWithLogitsLoss()
                    cross_loss = criterion(logits, labels.float())
                    contrastive_l = contrastive_loss(tem, embbedings.cpu().detach().numpy(), labels)
                    loss = (lam * contrastive_l) + (1 - lam) * (cross_loss)
                    

                # Backward pass
                loss.backward()
                
                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                # Esto viene del github de contrstive learning
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                # Update tqdm description with loss
                tepoch.set_postfix(loss=loss.item())
                
                #update learning rate
                if scheduler:
                    scheduler.step()

            # Print average loss for this epoch
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss}")
            train_losses += [avg_train_loss]
            
    return train_losses


def evaluate(model, test_dataloader, THRESHOLD=0.5, LOWER_UPPER_BOUND=False, val_or_test=None, plot_errors_distribution=False):
    test_outputs = []
    test_true_labels = []
    with torch.no_grad():
        total_test_loss = 0
        total_test_samples = 0
        for data in test_dataloader:
            input_ids, attention_mask, labels, _ = data
            # Save list of ground truth labels for metrics
            test_true_labels += labels.tolist()
                    
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
                    
            # Forward pass
            outputs = model(input_ids, attention_mask)
            test_outputs += outputs.tolist()
            logits = outputs.squeeze(-1)

            # Compute loss
            test_loss = criterion(logits, labels.float())

            # Accumulate test loss and total number of samples
            total_test_loss += test_loss.item() * labels.size(0)
            total_test_samples += labels.size(0)
            
    # Calculate average test loss
    average_test_loss = total_test_loss / total_test_samples   
                    
    test_predictions = [1 if x[0] > THRESHOLD else 0 for x in test_outputs]
    mcc = matthews_corrcoef(test_true_labels, test_predictions)
    print(f"{val_or_test} MCC {THRESHOLD} threshold:  {mcc:.4f}")
    
    # log in wandb
    if wandb.run is not None:
        wandb.log({f"{val_or_test}_{THRESHOLD}_threshold_MCC":mcc})

    ##### matriz de confusion
    cm = confusion_matrix(test_true_labels, test_predictions)
    print(cm)

    #### get index of errors in predictions
    errors = [i for i, x in enumerate(test_predictions) if x != test_true_labels[i]]
    #### get the model outputs for the errors
    errors_outputs = [test_outputs[i] for i in errors]
    #### plot the errors with title
    if plot_errors_distribution:
        plt.hist([x[0] for x in errors_outputs]) 
        plt.title(f"{val_or_test} errors distribution")
        plt.show()

    
    # Si hay valores para el rango de valor desconocido sobre-escribimos las predicciones  
    if LOWER_UPPER_BOUND != False:
        test_predictions = [1 if x[0] > LOWER_UPPER_BOUND[1] else (0 if x[0] <= LOWER_UPPER_BOUND[0] else -1) for x in test_outputs]
        mcc = matthews_corrcoef(test_true_labels, test_predictions)
        print(f"{val_or_test} MCC with Lower_bound:{LOWER_UPPER_BOUND[0]} and upper_bound:{LOWER_UPPER_BOUND[1]} > MCC: {mcc:.4f}")
        # log in wandb
        if wandb.run is not None:
            wandb.log({f"{val_or_test}_lower_upper_{LOWER_UPPER_BOUND[0]}-{LOWER_UPPER_BOUND[1]}_MCC":mcc})
            
    # Print average test loss for this epoch
    print(f"Average {val_or_test} Loss: {average_test_loss:.4f}")
    
    return test_outputs


def evaluate_kfold_ensemble(predictions, test_dataloader, THRESHOLD=0.5, LOWER_UPPER_BOUND=False):
    test_true_labels = []
    with torch.no_grad():
        for data in test_dataloader:
            input_ids, attention_mask, labels, _ = data
            # Save list of ground truth labels for metrics
            test_true_labels += labels.tolist()
            
    # Mean output
    test_outputs = np.array(predictions)
    test_outputs = test_outputs.mean(axis=0)
    test_mean_predictions = [1 if x > THRESHOLD else 0 for x in test_outputs]
    mcc = matthews_corrcoef(test_true_labels, test_mean_predictions)
    print(f"test ensemble MCC {THRESHOLD} threshold: {mcc:.4f}")
    # log in wandb
    if wandb.run is not None:
        wandb.log({f"test_ensemble_{THRESHOLD}_threshold_MCC":mcc})
    # Max Voting

    
# credit  -  https://github.com/sl-93/SUPERVISED-CONTRASTIVE-LEARNING-FOR-PRE-TRAINED-LANGUAGE-MODEL-FINE-TUNING/blob/main/main.py
def contrastive_loss(temp, embedding, label):
    """calculate the contrastive loss
    """
    # cosine similarity between embeddings
    cosine_sim = cosine_similarity(embedding, embedding)
    # remove diagonal elements from matrix
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    # calculate row sum
    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))
    # calculate outer sum
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        # calculate inner sum
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss