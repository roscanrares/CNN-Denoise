# DeepDenoiseCNN 
A deep learning-based approach for **image noise reduction** using Convolutional Neural Networks (CNNs).

##  Overview  
DeepDenoiseCNN is a machine learning model designed to remove noise from images by leveraging **deep learning techniques**.  
The model is trained on noisy-clean image pairs and learns to restore the original image quality. 
I developed a denoising algorithm for image preprocessing to enhance object recognition. The object recognition project is available [here](https://github.com/roscanrares/ObjectRecognition)

## Types of Noise Handled
	•	Gaussian Noise – Random pixel variations following a Gaussian distribution (commonly found in sensor noise).
	•	Salt & Pepper Noise – Random black and white pixels scattered in the image.
	•	Poisson Noise – Noise based on Poisson distribution, often appearing in low-light photography or medical imaging.


## How the model works
Preprocessing: The model loads images and adds artificial noise (if needed).
Training: The neural network learns to transform noisy images into clean ones.
Testing: The trained model is evaluated on new images to assess its denoising performance.
Comparison with traditional methods: Scikit-learn may be used to analyze results against classical denoising technique
