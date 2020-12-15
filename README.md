# plant_disease_classify

Project By: Sachit Nagpal (Net ID: sn2811)

Submission Documents: This README, along with [this](https://colab.research.google.com/drive/1jn06snZBrHpb07EQFON3Dx3hzQfXbjjg?usp=sharing) notebook contains the submission. 

## Project Purpose
The goal of the project is to use computer vision to accurately diagnose plant diseases from images that are clicked by farmers.

## Why Predict Plant Diseases? (About the Problem)
One of the challenges farmers face is crop loss due to plant diseases. Correctly identifying the disease early is crucial to preventing crop loss. Farmers have limited access to expert opinion and have to rely on their own knowledge and that of fellow farmers. It is estimated that nearly 50% of crop loss occurs due to plant diseases in developing countries such as India (**cite**). Hence, training these models can have significant impact on livelihood of farmers and food supply. Automated diagnosis can benefit particularly due to the widespread access to smartphones with HD cameras today. 

## Approach and Results
In this project, I train convolutional neural networks with transfer learning on two datasets (called [PlantVillage](https://arxiv.org/abs/1604.03169) and [PlantDoc](https://arxiv.org/abs/1911.10317)). I did not use any existing architectures (e.g. AlexNet, VGG, etc.). Instead, I stacked convolutional layers, max pool layers and finally some fully connected layers using the PyTorch library. For details on the exact settings, please refer to the [notebook](https://colab.research.google.com/drive/1jn06snZBrHpb07EQFON3Dx3hzQfXbjjg?usp=sharing). Further, I applied three stage transfer learning taking advantage of the convolutional implementation of sliding windows technique for the final model. The details of the three models and their results are below.

### Model 1
The first model is trained on PlantVillage dataset. 
It has an accuracy of 94% on the test set. 
This is similar to results in the [original paper](https://arxiv.org/abs/1604.03169) 
that introduced the dataset without pretraining. 
This model does not generalize to the PlantDoc dataset due to the synthetic nature of the dataset. 
Its accuracy on PlantDoc dataset is 11%. Even the authors of the PlantDoc dataset noted in their [paper](https://arxiv.org/abs/1911.10317) an accuracy of ~15%, when they trained a model on PlantVillage and tested on PlantDoc dataset. In it, they had pretrained on the ImageNet too which I did not do.  
### Model 2
The second model is trained on Cropped PlantDoc (C-PD) dataset (contains cropped leaves from images in PlantDoc datatset where cropping is done by creating rectangular shaped bounding boxes). 

Here, I apply transfer learning by initializing the weights from Model 1. It has an accuracy of 30% on the test set. On PlantDoc dataset, it has an accuracy of 18%. In the PlantDoc paper, it was shown that accuracy of ~70% is achievable with certain pretrained architectures on C-PD. I implemented a smaller network with no pretraining and achieve 30% accuracy on C-PD.

### Model 3
In model 3, I apply transfer learning again, but feed larger sized input image to the model. Thus, I implement the sliding window algorithm commonly used for Object Detection. Instead of detecting object though, I use the output for classification. By feeding a larger sized image, I get Model 2's predictions on smaller crops within the image. I add a dense hidden layer at the end of Model 2, followed by the output layer. The last two dense layers make a prediction for the entire image, using the predictions of model 2 on smaller cropped regions within the image. The idea is that Model 2 is expected to perform better in some of the cropped regions, instead of the entire image. Finally though, a prediction is needed for the entire image. Hence, I add two dense layers at the end whose goal is to learn how to classify the image based on Model 2's predictions.

For more specific details, can follow the [notebook](https://colab.research.google.com/drive/1jn06snZBrHpb07EQFON3Dx3hzQfXbjjg?usp=sharing).

### Summary of Results
Below are accuracies for the three models on the test sets of the three datasets.  

|Model/Test Set  | PlantVillage  | Cropped PlantDoc  | PlantDoc |
|---|---|---|---|
|Model 1    | 94%  | 10% | 11% |
|Model 2    | 19%  | 30% | 19% |
|Model 3    | 19%  | 18% | 21% |

Although Model 1 achieves high accuracy on PlantVillage, model 2 and 3 generalize more to real world images.

## Citing Prior Initial Work
In this project, I utilized some sections from my own prior work. In it, I had applied transfer learning in the same manner as here, but with VGG16 network pretrained on ImageNet dataset. The resulting accuracy was signficantly higher (~65% on PlantDoc) seemingly because of using VGG16 and pretraining on ImageNet. Model 1 and model 2 were also fed higher resolution images (100x100 instead of 64x64) which could be a reason. The code of my previous work is below:

- [Notebook 1](https://colab.research.google.com/drive/1oZXhQ0Hb7GdWOSW3NURywuevBNyAY-Yd?usp=sharing) 
- [Notebook 2](https://colab.research.google.com/drive/1I_uCu340A-RVZJwIn2rleWFjVinG6Yyq?usp=sharing)

The key additions from my prior work in this project can be summarized below - 

- I used a different CNN architecture. I stacked convolutional, max pool, and dense layers instead of using pretrained VGG16
- Model 3 here does not just take a larger sized image as input to implement the Sliding Window algorithm. It instead takes two larger sized images (one 256x256, and one 520x520) and concatenates the outputs of model 2 on both images before giving final prediction. The idea was to capture more cropped regions of different sizes this way. 
   
