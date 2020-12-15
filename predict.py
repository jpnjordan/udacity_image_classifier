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
import math

def main():
    start_time = time.time()
    
    in_arg = get_input_args()
    print("--------------------------------------------------------")
    print("Image: " + in_arg.image)
    print("Checkpoint: " + in_arg.checkpoint)
    print("Category Names: " + in_arg.category_names)
    print("Top K: " + str(in_arg.topK))
    print("GPU: " + str(in_arg.gpu))
    print("--------------------------------------------------------")
    
    model = load_checkpoint(in_arg.checkpoint, in_arg.gpu)
    
    image_path = in_arg.image
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(image_path, model, in_arg.topK, in_arg.gpu)
    print(classes)
    print(probs)
    for j in range(0,len(classes)):
        print("{}: {}  {:.2f}".format(j+1, cat_to_name[classes[j]].title(), (float(probs[j])*100)))
    
def load_checkpoint(filepath, gpu):
    #Amended 24/08/18 as suggested by assessor
    checkpoint = t.load(filepath)

    if checkpoint['arch'] == 'vgg16':
        model = tv.models.vgg16(pretrained =True)
    if checkpoint['arch'] == 'vgg13':
        model = tv.models.vgg13(pretrained =True)
    if checkpoint['arch'] == 'vgg13':
        model = tv.models.alexnet(pretrained=True)
    ordered_dict = OrderedDict([('fc1', t.nn.Linear(checkpoint['input_size'], checkpoint['hidden_units'])),
            ('relu', t.nn.ReLU()),
            ('dropout', t.nn.Dropout(p=checkpoint['fc1_dropout'])),
            ('fc2', t.nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
            ('output', t.nn.LogSoftmax(dim=1))])
    classifier = t.nn.Sequential(ordered_dict)
    model.classifier = classifier
    if t.cuda.is_available() and gpu:
        model.to('cuda')
    model.class_to_idx = checkpoint['class_to_idx']
    return model

#Amended 23/08/2018 as suggested by assessor
def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(PIL.Image.open(image_path))
    image = t.from_numpy(image)
    if gpu and t.cuda.is_available():
        image = image.cuda()
    image = image.unsqueeze_(0)

    model.eval()
    
    with t.no_grad():
        output = model.forward(image.float())
    ps = t.exp(output)
    probs, indices = ps.topk(5)
    class_names_dictionary = {k:v for v,k in model.class_to_idx.items()}
    indices_array = [] 
    for i in indices[0]:
        indices_array.append(i.item())
    classes = []
    for x in indices_array:
        classes.append(class_names_dictionary[x])
    probabilities = []
    for i in probs[0]:
        probabilities.append(i.item())
    return probabilities, classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #--------------
    #Amended 23/08/2018 with suggestions made by assessor
    width, height = image.size
    size = 256, 256
    if width > height:
        ratio = float(width) / float(height)
        newwidth = ratio * size[0]
        image = image.resize((int(math.floor(newwidth)), size[1]), PIL.Image.ANTIALIAS)
    else:
        ratio = float(height) / float(width)
        newheight = ratio * size[1]
        image = image.resize((size[0], int(math.floor(newheight))), PIL.Image.ANTIALIAS)
        

    image = image.crop((
        size[0] //2 - (224/2),
        size[1] //2 - (224/2),
        size[0] //2 + (224/2),
        size[1] //2 + (224/2))
    )

    #--------------
    #Code hint taken from slack from Tristan Newman
    image_array = n.array(image)/255.
    mean = n.array([0.485, 0.456, 0.406])
    std = n.array([0.229, 0.224, 0.225])
    normalized_image = (image_array - mean) / std  
    
    normalized_image = normalized_image.transpose((2, 0, 1))
    #--------------
    
    return normalized_image

def get_input_args():
    #TODO: Amend for this project
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, default='flowers/test/1/image_06743.jpg', 
                        help='image to predict')
    parser.add_argument('checkpoint', type=str, default='save_directory/checkpoint.pth', 
                        help='path to save checkpoint')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', 
                        help='chosen model')
    parser.add_argument('--topK', type=int, default=5,
                        help='learning rate')
    #Amended 24/08/18 Noticed it was always true. Founds solution: https://stackoverflow.com/questions/44561722/python3-why-in-argparse-a-true-is-always-true
    parser.add_argument('--gpu', default=True, action='store_false',
                        help='disable gpu')
    return parser.parse_args()

if __name__ == "__main__":
    main()