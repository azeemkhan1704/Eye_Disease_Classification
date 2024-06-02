### Eye Diseases Classification

This repository contains code for a deep learning model to classify different types of eye diseases based on images.

Dataset

The dataset used for this project can be found in the /dataset directory. It contains images of various eye diseases, including glaucoma, cataracts, diabetic retinopathy, and normal eyes.

Requirements
Python 3.x

TensorFlow 2.x

scikit-learn

matplotlib

pandas

numpy

OpenCV (cv2)

Models Used
1. Sequential Model
Accuracy: 86.84%
2. DenseNet121
Accuracy: 87.44%
3. MobileNetV2
Accuracy: 85.07%
4. VGG16
Accuracy: 85.07%
5. VGG19
Accuracy: 69.19%
6. ResNet50
Accuracy: 70.43%
7. Final Hybrid Model
Accuracy: 96.49% (Using GroupFold cross-validation)

The hybrid model combines the power of a pre-trained VGG19 convolutional neural network (CNN) with a custom sequential model.

Evaluation
The performance of each trained model is evaluated using the test dataset.
The evaluation includes classification reports and confusion matrices to assess the model's performance on different eye diseases.
