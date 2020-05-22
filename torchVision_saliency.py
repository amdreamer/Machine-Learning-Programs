import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import requests
from PIL import Image

# Using VGG-19 pretrained model for image classification
model_vision = torchvision.models.vgg19(pretrained=True)

# Imports and code for using pretrained VGG-19 model. 
# Note that since we don’t need to find gradients with respect to the parameters of the network, 
# so we’re setting param.requires_grad to False.
for param in model_vision.parameters():
    param.requires_grad = False

# download the image of the dog
def download(url,fname):
    response = requests.get(url)
    with open(fname,"wb") as f:
        f.write(response.content)
    
# Downloading the image    
download("https://specials-images.forbesimg.com/imageserve/5db4c7b464b49a0007e9dfac/960x0.jpg?fit=scale","input.jpg")

# Opening the image (mode = RGB, size = 960*640)
img = Image.open('input.jpg') 

# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

'''
    Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
    => Y/(1/σ) follows Distribution(0,σ)
    => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
'''
def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))

# preprocess the image
X = preprocess(img)
#  X.size() [1, 3, 224, 224]
# we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
X.requires_grad_()
# Saliency map is the gradient of the maximum score value with respect to the input image. 
# But note that the input image has 3 channels, R, G and B. To derive a single class saliency value for each pixel (i, j), 
# we take the maximum magnitude across all color channels. This can be implemented as follows.

# we would run the model in evaluation mode
model_vision.eval()
'''
forward pass through the model to get the scores, note that VGG-19 model doesn't perform softmax at the end
and we also don't need softmax, we need scores, so that's perfect for us.
'''
scores = model_vision(X) # scores.size() [1,1000] grad_fn=<AddmmBackward> 
# 1,000 is here because the last layer of the model is "Linear(in_features=4096, out_features=1000, bias=True)"
# this is a one-thousand classification model.

# Get the index corresponding to the maximum score and the maximum score itself.
score_max_index = scores.argmax() # index: 153
score_max = scores[0,score_max_index] # score: 21.8932

'''
backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
score_max with respect to nodes in the computation graph
'''
score_max.backward()
# Note: this backward function can be called only once. 
# For the second time: RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed
'''
Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
across all colour channels.
'''
saliency, _ = torch.max(X.grad.data.abs(),dim=1) 
# X.grad.data.abs(): [1, 3, 224, 224]
# saliency [1, 224, 224]

# code to plot the saliency map as a heatmap
plt.imshow(saliency.squeeze(), cmap=plt.cm.hot)
plt.axis('off')
plt.show()
plt.savefig('Saliency_for_dog.pdf', format='pdf')

# for original picture with 224*224
X_reshape = deprocess(X)
X_reshape.save('Picture_for_dog.pdf')
