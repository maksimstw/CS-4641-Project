## **Background**
In recent years, there has been a rising interest in electric cars and autopilots. As with any vehicle, an autonomous vehicle cannot merely follow the information on the map. It also needs to actively adapt to the changing road conditions. Most of the conditions are conveyed through road signs, and thus an autopilot system must be able to detect and understand road signs both quickly and accurately. This research project tries to implement the state-of-the-art computer vision algorithm residual networks (ResNet) and finetune it specifically for road sign detection. 

## **Problem Definition**
Given an image of a road sign, the deep learning model needs to recognize the road sign. Since this model is for an autopilot system, the deep learning model, ideally, should be able to detect and recognize the road sign quickly and accurately. 

## **Data**
The data was retrieved from Kaggle (https://www.kaggle.com/michaelcripman/road-sign-recognition). The dataset contained 46063 different road sign images. There were 182 different types of road signs. The images came in different shapes and resolutions. The dataset was split into 80% for training, 10% for validation, and 10% for testing. 

## **Methods**
### Models
The baseline model for this task was ResNet. We used Pytorch team's implementations of ResNet 18, ResNet 50, and ResNet 101. There were only 182 different classes in our custom dataset, so we modified the last fully connected layer of the original implementation. In addition, we implemented squeeze and excitation network to improve the performance of ResNet. 

### Data Preprocessing
Pytorch’s implementations of ResNet were originally based on ImageNet, which contained 1000 different image classes. All of the images had a resolution of 224x224. However, since all the road sign images came in different shapes and resolutions, they first need to be reshaped in order to be fed into ResNet properly. We first used transforms.RandomResizedCrop to randomly scale and crop the images to the proper shape. Next, we normalized the images to ensure our models converge properly and efficiently (by using mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). 

### Hyperparameters, optimizer, and loss function
|||
|---------|--------|
|Learning Rate|3e-4|
|Batch Size|64|
|Epochs|15|
|Optimizer|Adam|
|Loss Function| Cross entropy loss|

## ** Results and Discussion**


## **References**
Azizov, Said. "Road Sign Recognition." Kaggle. Retrieved June 13, 2021 from https://www.kaggle.com/michaelcripman/road-sign-recognition.  
Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).   
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.  
Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

## **Author**
| Name        | Email                |  
| ----------- | ---------------      |  
| Taiwei Shi  | maksimstw@gatech.edu |  
| Yitong Li   | yli3277@gatech.edu   |  
| Ruikang Li  | rakanli@gatech.edu   |  
| Xi Lin      | xlin315@gatech.edu   |
| Kelin Yu    | kyu85@gatech.edu     |
