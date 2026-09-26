# Evaluating Augmentation Reliability in Limited-Data Chest X-Ray Pneumonia Detection Using Grad-CAM

**Or Bakish and Yotam Gilboa**

## 1. Introduction and Related Work

Deep learning has shown strong performance in medical image classification, including the detection of pneumonia from chest X-ray images. However, training deep neural networks usually requires a large amount of labeled data. In medical imaging this can be a significant limitation, since collecting and labeling data may require expert knowledge and can be expensive. Therefore, it is useful to understand how well pretrained models can perform when only a limited amount of training data is available, and whether data augmentation can reduce the performance loss.

In this project, we study binary classification of chest X-rays into Normal and Pneumonia classes. Our main goal is to examine how training-data size and augmentation strength affect both classification performance and the stability of model explanations. We compare three pretrained CNN architectures of different sizes: MobileNetV3-Small, ResNet18, and EfficientNet-B0. Each model is trained using 20%, 50%, and 100% of the available training data, with no augmentation, mild medically realistic augmentation, or stronger augmentation. This gives a total of 27 main experimental configurations.

Previous work by Kermany et al. [1] demonstrated that deep learning can be successfully applied to pediatric chest X-rays for pneumonia classification. Their work also introduced the dataset used in this project and showed that transfer learning can achieve strong classification performance on this task. While classification accuracy is important, it does not show which image regions influence the model's decision. Selvaraju et al. [2] introduced Grad-CAM, a method for producing class-specific heatmaps that highlight image regions that contribute to a CNN prediction.

Based on these ideas, we use Grad-CAM not only for visualization but also as an auditing tool. We compare how stable the explanation maps remain under small input perturbations and across different training settings. This allows us to study whether augmentation improves classification performance while also affecting the consistency of the visual evidence used by the model. The project therefore focuses on the trade-off between available training data, augmentation strength, classification performance, and explanation stability.

## 2. Dataset

We use the publicly available Chest X-Ray Images (Pneumonia) dataset by Kermany et al. The dataset contains 5,856 pediatric chest X-ray images divided into two classes: Normal and Pneumonia. The original dataset contains 5,216 training images, 16 validation images, and 624 test images. The training set is imbalanced, with 3,875 Pneumonia images and 1,341 Normal images.

Because the original validation set contains only 16 images, it is too small for reliable model selection and early stopping. We therefore combine the original training and validation sets and create a new validation split of approximately 15%. The official test set is kept unchanged and is not used during model selection or hyperparameter tuning. When creating the new split, related images are kept together using filename-derived patient or case grouping where possible, in order to reduce the risk of leakage between training and validation.

Exploratory data analysis showed several dataset properties that may affect model behavior. Pneumonia images are generally smaller and have a higher aspect ratio than Normal images. In addition, 283 images are stored as RGB, and all of them belong to the Pneumonia class, while the remaining images are grayscale. To prevent the model from using channel format as a shortcut, all images are converted to grayscale and then replicated to three channels before being passed to the pretrained networks.

The analysis also identified other possible shortcuts, including differences in framing, image borders, laterality markers, text overlays, and medical devices such as tubes and electrodes. These differences are not necessarily related to pneumonia itself, but may still correlate with the class label. For this reason, they are considered when interpreting both classification results and Grad-CAM explanations.

All images are resized directly to 224×224 pixels. Aspect-preserving padding was avoided because the amount of padding would itself be correlated with the original aspect ratio and therefore with the class label. Direct resizing keeps the full image content, although the geometric distortion it introduces is treated as a limitation of the preprocessing pipeline.

## 3. Methodology

## 4. Results and Analysis

## 5. Discussion and Conclusion

## Team Contributions

## Code and Reproducibility

## References
