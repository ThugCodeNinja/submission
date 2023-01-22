from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from torch.utils import data
import argparse
import torch
from torch import nn
from torch import optim
import time
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='data_dir', type=str)
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='vgg_checkpoint.pth')
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def load_m(arch, hidden_units, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        num_in_features = 25088
    else:
        print("Invalid")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.to(device)
    return model, device, num_in_features


def train_model(epochs, traintfl, validtfl, model, device, criterion, optimizer):
    steps = 0
    running_loss = 0
    freq = 5

    start = time.time()
    print('Model undergoing Training...')

    for epoch in range(epochs):
        for inputs, labels in traintfl:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)  # Move input and label tensors to the default device

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % freq == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validtfl:
                        inputs, labels = inputs.to(device), labels.to(device)  # transfering tensors to the GPU

                        logps = model.forward(inputs)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / freq:.3f}.. "
                      f"Test loss: {test_loss / len(validtfl):.3f}.. "
                      f"Test accuracy: {accuracy / len(validtfl):.3f}")
                running_loss = 0
                model.train()

    end = time.time()
    total_time = end - start
    print(" Model Cmpleted Training in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))


def save_checkpoint(file_path, model, epochs, optimizer, learning_rate, input_size, output_size, arch, hidden_units):
    #model.class_to_idx = image_datasets[0].class_to_idx
    bundle = {
        'pretrained_model': arch,
        'input_size': input_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'classifier': model.classifier,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        #'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict()
    }

    torch.save(bundle, file_path)
    print("Model saved")

def main():
    print("Initializing")
    args = parse_args()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,                                     0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,                                     0.225])])
    test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,                                     0.225])])
    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transform)
    test_data = ImageFolder(root=test_dir, transform=test_transform)
    traintfl = data.DataLoader(train_data, batch_size=64, shuffle=True)
    validtfl = data.DataLoader(valid_data, batch_size=64)
    testtfl = data.DataLoader(test_data, batch_size=64)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # getting model, device object and number of input features
    model, device, num_in_features = load_m(args.arch, args.hidden_units, args.gpu)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(args.epochs, traintfl, validtfl, model, device, criterion, optimizer)

    file_path = args.save_dir

    output_size = 102
    save_checkpoint(file_path, model, args.epochs, optimizer, args.learning_rate,
                    num_in_features, output_size, args.arch, args.hidden_units)


if __name__ == "__main__":
    main()