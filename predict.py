import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms,models
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torch.utils import data
from PIL import Image
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import json


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = torchvision.models.vgg16(pretrained=True)
    
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    
def prepare_image(image_path):

    pil_image = Image.open(image_path)
    pil_image.resize((256,256))
    width, height = pil_image.size 
    new_width, new_height = 224, 224
    
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom
    pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2 , 0, 1))
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
    return tensor    

def predict(image_path, model, topk, device, cat_to_name):
    image = prepare_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    model.eval()
    #model.class_to_idx = image_datasets['train'].class_to_idx
    with torch.no_grad():
        ps = torch.exp(model(image))
        
    ps, top_classes = ps.topk(topk, dim=1)
    
    idx_to_flower = {v:cat_to_name[k] for k, v in image_datasets['train'].class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]
    return ps.tolist()[0], predicted_flowers_list

def print_pred(args):
    # load model
    model = load_checkpoint(args.model_filepath)
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    if args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
    else:
        device = 'cpu'

    model = model.to(device)
    # print(model.class_to_index)
    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)
    print("Predictions:")
    for i in range(args.top_k):
          print("{: <3} {: <25} Prob: {:.3f}%".format(i, top_classes[i], top_ps[i]*100))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='image_filepath', help="This is a image file that you want to classify")
    parser.add_argument(dest='model_filepath', help="This is file path of a checkpoint file, including the extension")
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', help="This is a file path to a json file that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="This is the number of most likely classes to return, default is 5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')

    args = parser.parse_args()
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(35),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}

    data_dir = 'flowers'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid','test']}


    dataset_sizes = {x: len(image_datasets[x])
                 for x in ['train', 'valid', 'test']}
    print_pred(args)