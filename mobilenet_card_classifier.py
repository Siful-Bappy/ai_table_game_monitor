import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
#%matplotlib inline
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

#### Check dataset ########################################
 
def check_card_dataset(dataset_dir):

    card_csv = pd.read_csv(os.path.join(dataset_dir, "cards.csv"))
    # change column name
    card_csv.columns = ["class", "filepaths", "labels", "card_type", "data_set"]

    if True:  # label info 
        #card_list = list(set(card_csv["labels"]))
        card_label_list = sorted(list(set(card_csv["labels"])))  # *** IMPORTANT *** KEEP ORDERING *** should sort !!!!!!
        with open('card_labels.json', 'w') as f:
            json.dump(card_label_list,  f, indent=4, ensure_ascii=False)

    # plot to see the card Image in category
    n = 0
    count = 0
    plot_idx = 1
    for i in range(len(card_csv)):
        if card_csv["class"].iloc[i] == n:
            # Joker has no spade, heart, dia, clover
            if card_csv["labels"].iloc[i] == "joker":
                n += 1
                continue

            path = os.path.join(dataset_dir, card_csv["filepaths"].iloc[i])
            img = Image.open(path)
            plt.subplot(13, 4, plot_idx)
            plt.imshow(img)
            plt.axis("off")
            n += 1
            count += 1
            plot_idx += 1
        if count == 4:
            count = 0
    plt.show()


###  Mobilenet ###############################################

import timm
def check_models():

    # Find mobilenetv1~ (which is pretrained)
    print(timm.list_models("mobilenetv1*", pretrained=True),"\n")
    # Decision model
    model = timm.create_model("mobilenetv1_125.ra4_e3600_r224_in1k", pretrained=True)

    # Check model information
    print(model.default_cfg, "\n") # Model default info
    print(model.global_pool, "\n") # Model global_pool
    print(model.get_classifier(), "\n") # In/Out Linear info


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("mobilenetv1_125.ra4_e3600_r224_in1k", num_classes = 53, global_pool = "avg", pretrained=True)
    
    def forward(self, x):
        return self.model(x)


# 2. Dataset, DataLoader

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, img_path, class_list, tf = None):
        self.transform = tf
        self.path = img_path
        self.card_class = class_list

    def get_image(self, filename):
        img = Image.open(filename).convert("RGB")
        img = self.transform(img)
        return img
    
    def __getitem__(self, idx):
        img_ = self.path[idx]
        img = self.get_image(img_)
        cls_name = os.path.basename(os.path.dirname(img_))
        label = self.card_class.index(cls_name)  ####  **** IT is Why SORT *****
        return img, label
    
    def __len__(self):
        return len(self.path)

# DataLoader (with device type)
class DeviceDL():
    def __init__(self, dl):
        self.dl = dl
        # If Cuda is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    def to_device(self, data):
        # If data is (list or tuple)
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        # Data to device(cuda)
        return data.to(self.device)
    
    def __iter__(self):
        for b in self.dl:
            yield self.to_device(b)

    def __len__(self):
        return len(self.dl)



### Train ###################################

# Model train tool
import torch.nn.functional as F
import pickle

class TrainHelper():

    def __init__(self, save_path = "./working/history.pickle", history=[]):
        self.history = history
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def accuracy(self, outputs, labels):
        pred = torch.argmax(outputs, dim=1)  # 예측 인덱스
        return torch.tensor(torch.sum(pred == labels).item() / len(pred))
        
    @torch.no_grad()
    def validation(self, model, batch):
        images, labels = batch
        out = model(images)
        acc = self.accuracy(out, labels)
        loss = F.cross_entropy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    @torch.no_grad()
    def evaluation(self, model, data_loader):
        model.eval()
        outputs = [self.validation(model, batch) for batch in data_loader]
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': round(epoch_loss.item(), 5), 'val_acc': round(epoch_acc.item(), 5)}

    def logging(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

        self.history.append(result)
        
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.history, f)


def train(checkpoint_path, dataset_dir):            

    # Labels # Is Label ok?
    with open("card_labels.json", "r", encoding="utf-8") as f:
         card_label_list = json.load(f)

    # 1. create a model  
    model = load_model(checkpoint_path, train_mode = True)
    
    # 2. data loader 
    # 2.1 path     
    train_img_paths = glob(os.path.join(dataset_dir, "train/*/*.jpg"))
    valid_img_paths = glob(os.path.join(dataset_dir, "valid/*/*.jpg"))
    # 2.2 tranformer 
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # 2.3 dataset
    train_dataset = MyDataset(train_img_paths, card_label_list, tf = train_transform)
    valid_dataset = MyDataset(valid_img_paths, card_label_list, tf = val_transform)
    # 2.4 dataloader
    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = 64, shuffle = True)
    train_dl = DeviceDL(train_dataloader)  
    valid_dl = DeviceDL(valid_dataloader)

    # 2. train 
    from tqdm import tqdm
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    val_acc_best = 0
    save_model_pth = "./working/"
    os.makedirs(save_model_pth, exist_ok=True)
    save_model_name = ""
    best_model_name = ""
    train_helper = TrainHelper()

    for epoch in range(epochs):
        model.train()
        train_losses = []

        # 2.1 Train
        for batch in tqdm(train_dl):
            img, label = batch
            outputs = model(img)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, label)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()

        # 2.2 Valid
        result = train_helper.evaluation(model, valid_dl)
        result["train_loss"] = torch.stack(train_losses).mean().item()

        # Save the best model
        if result["val_acc"] > val_acc_best:
            val_acc_best = result["val_acc"]
            save_model_name = os.path.join(save_model_pth, f"best_ep_{epoch}_{val_acc_best}.pt")
            best_model_name = save_model_name
            torch.save(model.state_dict(), save_model_name)
            print(f"Saved PyTorch Model State to {save_model_name}")

        train_helper.logging(epoch, result)
    
    # Test Result
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    '''
    test_img = glob("./test/*/*.jpg")
    test_dataset = MyDataset(test_img, card_list, tf = test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True) 
    test_dl = DeviceDL(test_dataloader)
 
    print("Test Check : ", train_helper.evaluation(model, test_dl))
    #save_model_name = os.path.join(save_model_pth, f"last_{val_acc_best}.pt")
    '''
    import shutil
    final_model_name = os.path.join(save_model_pth, f"best.pt")
    shutil.copy(best_model_name, final_model_name)
    

### Test All #############################################################
    
def test_all(model, dataset_dir, card_label_list):

   
    # 1. model
    '''
    # Define the device once
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

   
    save_model_pth = "./working/"
    checkpoint_path = os.path.join(save_model_pth, "best.pt")

    if not os.path.exists(checkpoint_path):
        # If running test() without prior train(), this path will be missing.
        raise ValueError(f"'{checkpoint_path}' is not a valid checkpoint path. Please run train() first.")

    # Instantiate the model
    model = MobileNet()
    
    # Load the state dictionary, mapping it to the defined device
    # This correctly handles loading a CUDA-trained model onto a CPU machine (or vice-versa)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    
    # Move the model itself to the defined device
    model.to(DEVICE)
    
    # Set model to evaluation mode
    model.eval()  # IMPORTANT for inference
    '''
    
    # 2. dataset
    test_transform =  load_image_preprocessor()
    '''
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    '''
    
    #test_img = glob("./test/*/*.jpg")
    test_img = glob(os.path.join(dataset_dir, "*/*.jpg"))
    test_dataset = MyDataset(test_img, card_label_list, tf = test_transform)
    # Ensure shuffle is False for consistent, reproducible testing
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = False)
    
    # DeviceDL handles moving the data to the correct device (based on torch.cuda.is_available()
    # which aligns with the DEVICE defined above)
    test_dl = DeviceDL(test_dataloader)
    
    # 3. Test Result
    test_helper = TrainHelper()
    
    # test_helper.evaluation expects the model and data to be on the correct device, 
    # which is handled by model.to(DEVICE) and DeviceDL.
    test_result = test_helper.evaluation(model, test_dl)
    
    print("Test Check : ", test_result)
    
    
### Export API ##################################################################

def load_model(checkpoint_path = None, train_mode = False):

    # 1. Define Device and Checkpoint Path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Run train() first.")
        return None

    # 3. Load Model
    # Instantiate the model (Ensure MobileNet class is available in scope)
    model = MobileNet() 
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    
    model.to(DEVICE)
    if not train_mode:
        model.eval()  # Set model to evaluation mode (CRITICAL)
    
    return model

def load_image_preprocessor():

    # 'test' transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return test_transform

    
def classify_card(model, preprocessor, image_or_path):

    """
    Tests a single image using the saved MobileNet model.

    Args:
        image_path (str): The file path to the image to test.
        #card_list (list): The sorted list of all class labels.

    Returns:
        str: The predicted card label.
    """

    #  Load and Transform Image
    device = next(model.parameters()).device
    if isinstance(image_or_path, str):
        try:
            img = Image.open(image_or_path).convert("RGB")
            # Apply transforms and add a batch dimension (1, C, H, W)
            input_tensor = preprocessor(img).unsqueeze(0).to(device)
        except FileNotFoundError:
            return f"Error: Image file not found at {image_or_path}"

    elif isinstance(image_or_path, np.ndarray):
        img_pil = Image.fromarray(image_or_path)
        input_tensor = preprocessor(img_pil).unsqueeze(0).to(device)

    else: # make sure it is a PIL
        img_pil = image_or_path
        device = next(model.parameters()).device
        input_tensor = preprocessor(img_pil).unsqueeze(0).to(device)

    # Prediction (single) : @TODO batch 
    with torch.no_grad():
        output = model(input_tensor)
        # Apply softmax to get probabilities (optional, but good practice)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # Get the index of the highest probability
        predicted_index = torch.argmax(probabilities, dim=1).item()
        #print(f"prob:{probabilities}")
    return predicted_index, probabilities[0,predicted_index]

def classify_cards(model, preprocessor, cards):

    """
    Tests a batch images using the saved MobileNet model.

    Args:
        cards: list of numpy cards  

    Returns:
        indices of cards and probs   

    """

    device = next(model.parameters()).device

    # Preprocess batch (each im is numpy HWC)
    batch = torch.stack([preprocessor(Image.fromarray(im)) for im in cards])
    batch = batch.to(device)

    # Inference
    with torch.no_grad():
        output = model(batch)                           # shape (N, num_classes)
        probs_all = torch.softmax(output, dim=1)        # convert to probabilities
        predicted_indices = torch.argmax(probs_all, dim=1)
        predicted_probs = torch.max(probs_all, dim=1).values

    # Return tensors or convert to lists
    return predicted_indices.cpu(), predicted_probs.cpu()


# Keep the original main block for execution flow:
if __name__ == "__main__":

    import sys

    if len(sys.argv) < 3:
        print("usage:python {sys.argv[0]} check/train/test file/directory") 

    if sys.argv[1] == "check":
        check_card_dataset(sys.argv[2])
        
    elif sys.argv[1] == "train":
        checkpoint_path = os.path.join("working", "best.pt")
        model = load_model(checkpoint_path, True)
        train(model, sys.argv[2])
         
    elif sys.argv[1] == "test" and os.path.isdir(sys.argv[2]):   
        checkpoint_path = os.path.join("working", "best.pt")
        model = load_model(checkpoint_path)
        with open("card_labels.json", "r", encoding="utf-8") as f:
            card_label_list = json.load(f)
        test_all(model, sys.argv[2], card_label_list)
    elif sys.argv[1] == "test" and not os.path.isdir(sys.argv[2]) and sys.argv[2][-3:] in ["jpg", "png"] :   
    
        with open("card_labels.json", "r", encoding="utf-8") as f:
            card_label_list = json.load(f)

        #checkpoint_path = os.path.join("working", "best.pt")
        checkpoint_path = "mobilenet_card.pt"
        model = load_model(checkpoint_path)
        print("model type:", type(model.model))
        preprocessor  = load_image_preprocessor()
        if model is None or preprocessor is None:
            print("cannot load model or transformer")
            exit()

        single_image_path = sys.argv[2]  #single_image_path = "./test/ace of spades/1.jpg"
        if False: # using file path 
            idx_pred, prob = classify_card(model, preprocessor, single_image_path)
        else:
          
            single_image = cv2.imread(single_image_path)
            idx_pred, prob = classify_card(model, preprocessor, single_image)
        label_pred = card_label_list[idx_pred] # Use the predicted index to look up the label in the (sorted) card_list
        print(f"\nImage: {single_image_path}: Prediction: {label_pred}")

        img = cv2.imread(single_image_path)
        cv2.imshow("card", np.array(img))
        cv2.waitKey(0)
        #img.show()


     # TODO : use numpy image for classification 


