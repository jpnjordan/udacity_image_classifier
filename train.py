# PROGRAMMER: Jason Jordan
# DATE CREATED: 20/08/2018                                  
# PURPOSE: Train a given nueral network & data set. 
import argparse
from os import listdir
from time import time
import json
import numpy as n
import pandas as p
import torch as t
import torchvision as tv
import time
import PIL
from collections import OrderedDict

def main():
    start_time = time.time()
    
    in_arg = get_input_args()
    print("--------------------------------------------------------")
    print("Directory: " + in_arg.data_directory)
    print("Save Directory: " + in_arg.save_dir)
    print("Architecture: " + in_arg.arch)
    print("Learning Rate: " + str(in_arg.learning_rate))
    print("Hidden Units: " + str(in_arg.hidden_units))
    print("Epochs: " + str(in_arg.epochs))
    print("GPU: " + str(in_arg.gpu))
    print("--------------------------------------------------------")
    
    valid_dir = in_arg.data_directory + 'valid/'
    train_dir = in_arg.data_directory + 'train/'
    
    training_transformations = tv.transforms.Compose([tv.transforms.RandomRotation(40),
                                        tv.transforms.RandomResizedCrop(224),
                                        tv.transforms.RandomHorizontalFlip(),
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    validation_transformations = tv.transforms.Compose([tv.transforms.Resize(256),
                                        tv.transforms.CenterCrop(224),
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_set = tv.datasets.ImageFolder(train_dir, transform=training_transformations)
    validation_set = tv.datasets.ImageFolder(valid_dir, transform=validation_transformations)
    train_loader = t.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    validation_loader = t.utils.data.DataLoader(validation_set, batch_size=64)
    
    if in_arg.arch == "vgg13":
        model = tv.models.vgg13(pretrained=True)
        input_size = 25088
    elif in_arg.arch == "alexnet":
        model = tv.models.alexnet(pretrained=True)
        input_size = 9216
    else:
        model = tv.models.vgg16(pretrained=True)
        input_size = 25088
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = t.nn.Sequential(OrderedDict([
                              ('fc1', t.nn.Linear(input_size, in_arg.hidden_units)),
                              ('relu', t.nn.ReLU()),
                              ('dropout', t.nn.Dropout(p=0.2)),
                              ('fc2', t.nn.Linear(in_arg.hidden_units, 102)),
                              ('output', t.nn.LogSoftmax(dim=1))
                              ]))

    # Support architechture
    criterion = t.nn.NLLLoss()
    optimizer = t.optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    # Training parameters
    epochs = in_arg.epochs
    print_every = 50
    steps = 0   
    
    print('Training Started. Please stand by...')
    tt = time.time() 
    if t.cuda.is_available() and in_arg.gpu:
        model.to('cuda')
    else:
        model.to('cpu')
    model.train()

    for e in range(0,epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
                
            if t.cuda.is_available() and in_arg.gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with t.no_grad():
                    test_loss, accuracy = validation(model, validation_loader, criterion, in_arg.gpu)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validation_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))

                running_loss = 0
                model.train()
    
    model.class_to_idx = train_set.class_to_idx
    checkpoint = {'arch': 'vgg16',
                  'criterion': criterion,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'hidden_units': in_arg.hidden_units, 
                  'input_size': input_size,
                  'fc1_dropout': 0.2,
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx
                 }

    t.save(checkpoint, in_arg.save_dir + 'checkpoint.pth')
    # Measure total program runtime by collecting end time
    end_time = time.time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )    
    
    
def get_input_args():
    #TODO: Amend for this project
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, default='flowers/', 
                        help='training directory')
    parser.add_argument('--save_dir', type=str, default='save_directory/', 
                        help='path to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg13', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024,
                        help='hidden units')
    parser.add_argument('--epochs', type=int, default=5,
                        help='epochs')
    #Amended 24/08/18 Noticed it was always true. Founds solution: https://stackoverflow.com/questions/44561722/python3-why-in-argparse-a-true-is-always-true
    parser.add_argument('--gpu', default=True, action='store_false',
                        help='disables gpu')
    return parser.parse_args()

def validation(model, validation_loader, criterion, gpu):
    test_loss = 0
    accuracy = 0
    for inputs, labels in validation_loader:
        if gpu and t.cuda.is_available():
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        else:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = t.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(t.FloatTensor).mean()
    
    return test_loss, accuracy

if __name__ == "__main__":
    main()