# -Mycobacterium-Tuberculosis-Detection-Using-CNN

## Abstract

This research work is based on the various experiments performed for the detection of lung
tuberculosis using various methods like filtering, segmentation, feature extraction and
classification. Tuberculosis (TB) is a global issue that seriously endangers public health. Pathology
is one of the most important means for diagnosing TB in clinical practice. To confirm TB as the
diagnosis, finding specially stained TB bacilli under a microscope is critical. Because of the very
small size and number of bacilli, it is a time-consuming and strenuous work even for experienced
pathologists, and this strenuosity often leads to low detection rate and false diagnoses. We
investigated the clinical efficacy of an artificial intelligence (AI)-assisted detection method for
classification of X-ray images into Normal and Tuberculosis. We built a convolutional neural
networks (CNN) model, named tuberculosis AI (TB-AI), specifically to recognize Mycobacterium
Tuberculosis. The training set contains 1235 samples, including 1000 positive cases and 235
negative cases, where bacilli are labeled by human pathologists. Upon training the neural network
model. We compared the diagnosis of TB-AI to the ground truth result provided by human
pathologists, analyzed inconsistencies between AI and human, and adjusted the protocol
accordingly. Trained TB-AI were run on the test data twice. And we deploy the CNN model using
flask method so it can used anywhere and it is available for everyone around the world. 

# Introduction:

 Tuberculosis is an infectious disease that is usually caused in the lungs and the 2nd
deadliest disease. So, it makes it necessary for its detection to be as early as possible.
Recognition of bacteria is very significant in order to avert diseases and maintain the physical
condition of the world’s population. Mycobacterium tuberculosis is one of the perilous
bacteria which can be lethal to humans, children, and an animal's life. It caused by bacteria
thus their diagnosis is mainly based on sputum examination which is very tedious and timetaking.
 Nowadays DIP is extensively utilized in the medical domain for identifying various
category of malady and disorder. Chest X-ray (CXR) plays a crucial role in TB diagnosis,
especially pulmonary TB (PTB),which is one of the most common presentations of TB.
In addition, since CXRs provide a low cost, rapid examination even in remote settings, it has
been recognized as a powerful screening test for TB, especially in areas and populations with
higher disease burden. While the cost of acquiring a CXR had become much more
affordable, the interpretation of CXR scans is currently limited by cost and access to trained
radiologists. And many patients are diagnosed too late, being unable to treat their symptoms
using conventional TB antibiotics. Hence the importance of creating models that can analyze
a CXR.
There are two types of TB:
Latent TB: A person can have TB bacteria but doesn’t develop the disease. In this case there
are not any abnormalities in the chest X-rays (CXR).
Active TB: If the immune system can’t stop the bacteria from growing then the person shows
symptoms that can be seen in chest X-rays (CXR). 

# Problem Statement:

 Tuberculosis (TB) is a transferable malady caused by Mycobacterium Tuberculosis
(MTB) and invades when the infected people sneeze, cough and speak without mask. The
germs can exist in the air for some hours that consequence persons who breathe in the air
may become infected that mainly influence the lungs; it also influenced other limbs of the
body such as spine, kidneys and brain. TB malady is very common due to the weakening of
the immune system and the possibility of patient’s death enhances with time if left undiagnosed.
 Conventional TB diagnosis requires much time and money because of microscopic
examination of sputum, so automatic recognition is more valuable to prevent serious
consequences rather than manually. In this framework, we present an efficient approach for
automatic identification of Tuberculosis using some image processing techniques. So, we
can detect Mycobacterium Tuberculosis early and diagnose it as early as possible.

# Objective:

1. Collection of datasets of X-ray images of chest from Kaggle.
2. Developing an algorithm to train our proposed model.
3. Developing an algorithm using CNN for feature extraction and classification of
images.
4. Observing the accuracy of the proposed method and comparing with the other
existing method. 

# Methodology: 

Image Preprocessing
 The data set we have is not uniform and does not have fine textural features. Therefore,
this is the first step in image processing. This is done to get uniformity in the complete
dataset. This also enhances the various changes in the images and the different regions have
a much higher chance of getting detected. Median filtering is applied, for enhancing the
image. This method is to mainly focus on the noise removal operations from the image.
Image Segmentation
 In this section, the images with similar pixel values get grouped together, forming
regions. Segmentation is accurate when five qualities are taken care of.
Completeness: every pixel should belong to a region.
Connectedness: the points of region should be connected with some reason.
Disjointedness: There should be some property which differentiates each region.
Satisfiability: Pixel of a region must have at least one of the property.
Segmentability: Two regions should not be merged as one as they have different properties.
Feature Extraction
 The feature extraction technique helps to identify the features in the given image.
Regional properties area, major axis, minor axis and eccentricity is calculated. Statistical features like mean, standard deviation, kurtosis and skewness are also calculated.
Classification
 Convolutional Neural Networks (CNN) are used for pattern recognition in images. The
input of the network is an image of size m × m × r, where m is the height and width of the
image and r is the number of channels.
They consist of three types of layers:
Convolutional layers: Every layer has k filters (named kernels) of size n × n × q, where n is
smaller than m, giving as output k feature maps of size m − n + 1.
Pooling layers: Each map is down sampled in this layer, reducing the number of features.
Fully connected layers: They perform the usual job of a multi-layer neural network. In this
case binary classification. 

# Classification accuracy:

**Previous Method**
- KNN - 80%
- SMO - 75%
- Simple Linear Regression - 79%

**Proposed Method**
- CNN - 94.97%

# Conclusion:

This work presents a transfer learning approach with deep Convolutional Neural Networks
for the automatic detection of tuberculosis from the chest radiographs with accuracy of
94.97%. This work can be a very useful and fast diagnostic tool, which can save significant
number of people who died every year due to delayed or improper diagnosis. The CNN
model trained is deployed into the website using flask, Flask is used to deploy a machine
learning model into a web which can be accessible through website without any libraries .
This will help people to use themselves to predict the results and its also a time saving
method. Further it can develop in worldwide with increased accuracy. 
