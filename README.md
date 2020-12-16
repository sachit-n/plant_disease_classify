# Classifying Crop Diseases

Project Members: 
- Sachit Nagpal, Net ID - sn2811


Submission Documents: 
- This README contains the summary of the project executed. 
- The documentated code [here](https://colab.research.google.com/drive/1jn06snZBrHpb07EQFON3Dx3hzQfXbjjg?usp=sharing) may be referred too for specific details on implementation and/or reproducing results. 

## Project Purpose
The goal of the project is to use computer vision to accurately diagnose plant diseases from images that are clicked by farmers.

## Why Predict Plant Diseases? (About the Problem)
One of the challenges farmers face is crop loss due to plant diseases. Correctly identifying the disease early is crucial to preventing crop loss. Farmers have limited access to expert opinion and have to rely on their own knowledge and that of fellow farmers. It is estimated that nearly 50% of crop loss occurs due to plant diseases in developing countries such as India (in various reports, e.g. [this](https://pubmed.ncbi.nlm.nih.gov/24535397/)). Hence, training these models can have significant impact on livelihood of farmers and food supply. Automated diagnosis can benefit particularly due to the widespread access to smartphones with HD cameras today. 

## Approach and Results
In this project, I train convolutional neural networks with transfer learning on two datasets (called [PlantVillage](https://arxiv.org/abs/1604.03169) and [PlantDoc](https://arxiv.org/abs/1911.10317)). I did not use any predefined architectures (e.g. AlexNet, VGG, etc.). Instead, I stacked convolutional layers, max pool layers and some fully connected layers a certain way using PyTorch library. PyTorch was used because it was the library used in the prior initial work (cited at the end). For the entire code, please refer to the [notebook](https://colab.research.google.com/drive/1jn06snZBrHpb07EQFON3Dx3hzQfXbjjg?usp=sharing). Further, I applied three stage transfer learning taking advantage of the convolutional implementation of sliding windows technique for the final model. The details of the three models and their results are below.

### Model 1
Model 1 is trained on PlantVillage dataset. It has ~55k images, but the images are taken in controlled settings and model does not generalize to real world dataset.
The CNN architecture I used achieved around 94% accuracy on the test set of PlantVillage. 
This is similar to results in the [original paper](https://arxiv.org/abs/1604.03169) 
that introduced the dataset. When tested on a dataset that contains images in natural settings ([PlantDoc]([PlantDoc](https://arxiv.org/abs/1911.10317))), the accuracy was only ~11%.
The authors of the PlantDoc dataset noted in their paper an accuracy of ~15%, when they trained a model on PlantVillage and tested on PlantDoc dataset. I used a much smaller network than them and did not use pretrained weigths from ImageNet like them but and achieve ~11% accuracy.  

### Model 2
I train the second model by initializing the weights after training on PlantVillage dataset (i.e. model 1). The training is done on a dataset derived by cropping out the leaf portions by making rectangular boxes from the PlantDoc dataset. The authors of it call it the Cropped PlantDoc dataset and I will refer it as C-PD in future. The accuracy estimate measured increased to ~18% (from ~11%) on PlantDoc dataset. On the test split of C-PD, it had an accuracy of ~30%. Although the accuracy on PlantVillage reduced to 19%, it is still a better model since it generalizes better. 

### Model 3
In Model 3, I change the architecture partially. The idea was taken from my prior initial work (cited in the end). I add two dense layers at the end keeping the rest of the architecture to be the same with weights from Model 2. I then feed 2 versions of the image, one of size 520 x 520, and other of 256 x 256. Since the original models 1 and 2 were fed 64x64 sized images, the outputs I get are Model 2 predictions on 64x64 sliding window grids with stride of 6 for each image (256x256 one and 520x520 version). Due to two images, there are more rectangular grids whose predictions are used by the last two dense layers. The idea was inspired from results in the original PlantDoc paper where they were able to train a fairly accurate model for C-PD (~70%) but not PlantDoc (~29%). Hence, an accurate model on C-PD, may do better on cropped regions within PlantDoc dataset image than the entire image. Such predictions can then be utilized by another classifier to give final prediction for the whole image. In this case the final classifier are the last two dense layers. For first roound, I freezed all weights besides last two layers and retrained the whole network in the second round.

For detailed implementation settings, please refer the [notebook](https://colab.research.google.com/drive/1jn06snZBrHpb07EQFON3Dx3hzQfXbjjg?usp=sharing).

### Summary of Results
Below are accuracies for the three models on the test sets of the three datasets.  

|Model/Test Set  | PlantVillage  | Cropped PlantDoc  | PlantDoc |
|---|---|---|---|
|Model 1    | 94%  | 10% | 11% |
|Model 2    | 19%  | 30% | 19% |
|Model 3    | 19%  | 18% | 21% |

## Citing Prior Initial Work
In this project, I utilized some sections from my own prior work. In it, I had applied transfer learning in similar manner as here. In it, I had pretrained the model (VGG16) on ImageNet and achieved a significantly higher accuracy (~65% on PlantDoc). The code of my previous work is below:

- [Notebook 1](https://colab.research.google.com/drive/1oZXhQ0Hb7GdWOSW3NURywuevBNyAY-Yd?usp=sharing) 
- [Notebook 2](https://colab.research.google.com/drive/1I_uCu340A-RVZJwIn2rleWFjVinG6Yyq?usp=sharing)

The key additions from my prior work in this project can be summarized below - 

- I used a different CNN architecture. I stacked convolutional, max pool, and dense layers in a certain way instead of using pretrained VGG16. 
- The more prominent change was using two larger sized images for Model 3 to capture more cropped regions. The output on both images undergoes max pooling first, concatenated next before being fed to the last two dense layers. In my prior work, only a single larger sized image was fed to the model. 
   
