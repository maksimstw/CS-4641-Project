# Georgia Tech CS 4641 Project
## **Background**
In recent years, there has been a rising interest in electric cars and autopilots. As with any vehicle, an autonomous vehicle cannot merely follow the information on the map. It also needs to actively adapt to the changing road conditions. Most of the conditions are conveyed through road signs, and thus an autopilot system must be able to detect and understand road signs fast and accurately. This research project tries to implement the state-of-the-art compute vision algorithms, namely, vision transformer (ViT) and residual network (ResNet), and finetune them specifically for road sign detection. 

## **Problem Definition**
Given an image of a road sign, the deep learning model needs to recognize the road sign. Since this model is for an autopilot system, the deep learning model, ideally, should be able to detect and recognize the road sign quickly and accurately. 

## **Data**
The data is retrieved from Kaggle (https://www.kaggle.com/michaelcripman/road-sign-recognition). The dataset contains 5152 different road sign images. The images come in different shapes and resolutions. The dataset is split into 80% for training, 10% for validation, and 10% for testing. 

## **Methods**
Since all the images have different shapes and resolutions, they need to be reshaped or set paddings. The baseline model for this task would be ResNet. We would try to improve the model by implementing attention such as squeeze and excitation network (SENet). Lastly, we would also love to explore the newly developed computer vision model vision transformer (ViT) on this task. It is important to understand which task transformers are better than convolution neural networks and the reasons behind them. 

## **Potential Results**
ResNet performs incredibly well in most of the computer vision tasks, so we expect it to perform exceptionally well on this task also. With attention mechanisms such as SENet added, it is supposed to perform even better in terms of accuracy. However, since this model needs to be implemented on an autopilot system, it also needs to be fast. In the past, we have to find a balance and tradeoff between efficiency and accuracy, but this might no longer be the case due to the invention of transformer. We are excited to see if transformer could perform both efficiently and accurately. 

## **References**
Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

## **Getting Started**
### **Dependencies**
* Python 3.5 or higher
* PyTorch
* numpy

### **Executing program**
Run main.py

## **Author**
| Name        | Email                |
| ----------- | ---------------      |
| Taiwei Shi  | maksimstw@gatech.edu |
| Yitong Li   |                      |
| Ruikang Li  |                      |
| Xi Lin      |                      |
| Kelin Yu    |                      |
