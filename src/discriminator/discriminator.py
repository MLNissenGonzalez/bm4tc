# Building the discriminator class out of a predefined MPS. 
import torch
from torch.utils.data import DataLoader
from typing import Union, Sequence, Callable, Dict

# TODO: Add the discriminator class taking an MPS as input
#       and returning a pytorch module that is the MPS with an
#       MLP at the end to discriminate real from fake inputs to the MPS


# TODO: Classical discriminator initialisation based on predefined MPS or data.
# TODO: Optimizer initialisation
# TODO: Add discriminator ensemble, one for each class. 


#------------------------------------------------------------------------------------------------´
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#--------------------DISCRIMINIATOR PRETRAINING----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# TODO: Add documentation
# TODO: Allow for other losses
bce_loss = torch.nn.BCELoss() 
def discriminator_pretraining(dis,
                              max_epoch: int,
                              patience: int,
                              optimizer: torch.optim.Optimizer,
                              loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                              loader_train: DataLoader,
                              loader_test: DataLoader):
        
    train_loss = []
    test_loss = []
    test_accuracy = []
    patience_counter = 0
    best_loss = float('inf')
    optimizer.zero_grad()
    for i in range(max_epoch):
        dis.train()
        for X, t in loader_train:
            logit = dis(X)
            prob = torch.sigmoid(logit.squeeze())
            loss = loss_fn(prob, t.float())
            train_loss.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        dis.eval()
        total_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, t in loader_test:
                logit = dis(X)
                prob = torch.sigmoid(logit.squeeze())
                preds = (prob >= 0.5).long()
                loss = bce_loss(prob, t.float())
                total_test_loss += loss.item() * t.size(0)
                correct += (preds == t).sum().item()
                total += t.size(0)

            avg_test_loss = total_test_loss / total
            acc = correct / total
            test_loss.append(avg_test_loss)
            test_accuracy.append(acc)

            # Progress tracking and best model update
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                patience_counter = 0
                best_model_state = dis.state_dict()
            else:
                patience_counter += 1
            # Early stopping
            if patience_counter > patience:
                break

    dis.load_state_dict(best_model_state)
    
    return dis, train_loss, test_loss, test_accuracy, i