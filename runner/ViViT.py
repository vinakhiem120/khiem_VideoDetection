import torch
from torch import nn
import av
import numpy as np
from transformers import VivitImageProcessor, VivitModel
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import utils
from timeit import default_timer as timer
device = utils.get_device()

class runner():
    def __init__(self,config, logger):
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.image_size = config.data.image_size 
        self.num_output = config.data.num_output
        
        self.train_path = str(config.path.train_path)
        self.test_path = str(config.path.test_path)
        self.lr = config.train.lr
        
        self.epoch = config.train.epoch
        self.criterion = nn.CrossEntropyLoss()
        self.configuration = VivitConfig(num_frames = int(25*5))
        self.model = VivitModel(self.configuration)
        self.optimizer = self._get_optim(config.train.optimizer)
        
        self.transform = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self._init_data()
        self.get_device()
    def get_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device
    def _init_data(self):
        pass 
        
    def _get_optim(self,optim):
        if str(optim) == "Adam":
            return Adam(self.model.parameters(),lr=self.lr)
        return None
    def train_step(self,
                    model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    loss_fn: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer):
        # Put model in train mode
        model.train()
        
        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0
        
        # Loop through data loader data batches
        for batch, (X,y) in enumerate(dataloader):
            # Send data to target device
            X,y = X.to(self.device), y.to(self.device)

            # Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    def test_step(self,model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module):
        # Put model in eval mode
        model.eval() 
        
        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        
        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X,y) in enumerate(dataloader):
                # Send data to target device
                X,y = X.to(device), y.to(device)
        
                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                
                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                
        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc
    def train_model(self,model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
        
        # 2. Create empty results dictionary
        results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        
        # 3. Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer)
            test_loss, test_acc = self.test_step(model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn)
            
            # 4. Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            # 5. Update results dictionary1
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # 6. Return the filled results at the end of the epochs
        return results
    def train(self):
        self.model = self.model.to(self.device)
        start_time = timer()
        model_results = self.train_model(model=self.model, 
                        train_dataloader=self.train_loader,
                        test_dataloader=self.test_loader,
                        optimizer=self.optimizer,
                        loss_fn=self.criterion, 
                        epochs=self.epoch)
        
        end_time = timer()
        print(f"Total training time: {end_time - start_time:.3f} seconds")
        
        
        