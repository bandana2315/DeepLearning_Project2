                        **************************HANDWITTEN DIGIT RECOGNITION USING DEEP LEARNING*********************************


 Introduction:

 It is easy for the human brain to process images and analyse them. When the eye sees a certain   image, the brain can easily segment it and recognize its different elements. 
There is a field in computer science that tries to do the same thing for machines, which is Image Processing. The image to be processed is imported then analysed using some 
computations, which, by the end, results either in an image with a better quality or some of the characteristics of this image depending on the purpose of this analysis. 
 In this project, we will focus on building a mechanism that will recognize handwritten digits. We will be reading images containing handwritten digits extracted from the 
MNIST database and try to recognize which digit is represented by that image. For that we will use basic Image Correlation techniques, also referred to as Matrix Matching. 
This approach is based on matrices manipulations, as it reads the images as matrices in which each element is a pixel. It 2 overlaps the image with all the images in
 the reference set and find the correlation between them in order to be able to determine the digit it represents. 
The goal of this project is to apply and manipulate the basic image correlation techniques to build program and keep polishing and enhancing in order to
 investigate to which extent it can get improved. This would allow us to see how far we can go, in terms of accuracy and performance, but using just the very
 simple and basic techniques of matrix matching and without going into complicated methods like machine learning.

Motivation:
   
Handwritten digit recognition is the process to provide the ability to machines to recognize human handwritten digits.
 It is not an easy task for the machine because handwritten digits are not perfect, vary from person-to-person, and can be made with many different flavors.

Objective:

 Handwritten digit recognition using deep learning.

Problem Statement:

In this project, I use the MNIST dataset for the implementation of a handwritten digit recognition system. 
To implement this I used a special type of deep neural network called Convolutional  Neural Networks. Network (CNN). 
CNN is an artificial neural network that has the ability to detect patterns in the images. 
In the end, I build a Graphical user interface (GUI) where you can directly draw the digit and recognize it straight away.


Software requirements:

Language used: Python 3.7 Operating system: Windows10
Tool used: Anaconda, Jupyter Notebook

Hardware Requirements:

	Processor: Intel core i5 or above.
	64-bit, quad-core, 2.5 GHz minimum per core
	Ram: 4 GB or more
	Hard disk: 10 GB of available space or more.
	Display: Dual XGA (1024 x 768) or higher resolution monitors
	Operating system: Windows

Technology Used
Python- It is an interpreted , high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, 
Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.
Python is often described as a "batteries included" language due to its comprehensive standard library.

Deep learning!

Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. 
These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing it to “learn” from large amounts of data. 
While a neural network with a single layer can still make approximate predictions, additional hidden layers can help to optimize and refine for accuracy.

Deep learning drives many artificial intelligence (AI) applications and services that improve automation, performing analytical and physical tasks without human intervention.
Deep learning technology lies behind everyday products and services (such as digital assistants, voice-enabled TV remotes, and credit card fraud detection) as well as emerging 
technologies (such as self-driving cars).



Algorithm Used:

Convolutional neural networks(ConvNet/CNN): 

A Convolutional Neural Network is a Deep Learning algorithm which can take in an input image, assign importance to various aspects/objects in the image and be able to 
differentiate one from the other. Thepre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters 
are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex.
 Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. 
A collection of such fields overlap to cover the entire visual area.
Convolutional neural networks are distinguished from other neural networks by their superior performance with image, speech, or audio signal inputs.
 They have three main types of layers, which are:

Convolutional layer-

	Convolutional Layer
	Pooling layer
	Fully-connected (FC) layer

The convolutional layer is the core building block of a CNN, and it is where the majority of computation occurs.
 It requires a few components, which are input data, a filter, and a feature map. Let’s assume that the input will be a color image, 
which is made up of a matrix of pixels in 3D. This means that the input will have three dimensions—a height, width, and depth—which correspond to RGB in an image.
 We also have a feature detector, also known as a kernel or a filter, which will move across the receptive fields of the image, checking if the feature is present. 
This process is known as a convolution.

Pooling Layer:

Pooling layers, also known as downsampling, conducts dimensionality reduction, reducing the number of parameters in the input. 
Similar to the convolutional layer, the pooling operation sweeps a filter across the entire input, but the difference is that this filter does not have any weights. 
Instead, the kernel applies an aggregation function to the values within the receptive field, populating the output array. There are two main types of pooling:


Max pooling:

 As the filter moves across the input, it selects the pixel with the maximum value to send to the output array. 
As an aside, this approach tends to be used more often compared to average pooling.

Average pooling:

 As the filter moves across the input, it calculates the average value within the receptive field to send to the output array.
While a lot of information is lost in the pooling layer, it also has a number of benefits to the CNN. They help to reduce complexity, 
improve efficiency, and limit risk of overfitting.

Fully-Connected Layer:

The name of the full-connected layer aptly describes itself. As mentioned earlier, the pixel values of the input image are not directly connected to the output 
layer in partially connected layers. However, in the fully-connected layer, each node in the output layer connects directly to a node in the previous layer.
This layer performs the task of classification based on the features extracted through the previous layers and their different filters. 
While convolutional and pooling layers tend to use ReLu functions, FC layers usually leverage a softmax activation function to classify inputs appropriately, 
producing a probability from 0 to 1.
 

Types of convolutional neural networks:

	AlexNet
	VGGNet
	GoogLeNet
	ResNet
	ZFNet

However, LeNet-5 is known as the classic CNN architecture.


Some common applications of CNN:

Marketing: Social media platforms provide suggestions on who might be in photograph that has been posted on a profile, making it easier to tag friends in photo albums.

Healthcare: Computer vision has been incorporated into radiology technology, enabling doctors to better identify cancerous tumors in healthy anatomy.

Retail: Visual search has been incorporated into some e-commerce platforms, allowing brands to recommend items that would complement an existing wardrobe.

Automotive: While the age of driverless cars hasn’t quite emerged, the underlying technology has started to make its way into automobiles, 
improving driver and passenger safety through features like lane line detection.
 
Process involved:

1.	Import libraries and dataset

At the project beginning, import all the needed modules or libraries for training our model like Pandas, NumPy, Matplotlib, Tensorflow and easily read the 
training and testing dataset from the loaded file train.csv and test.csv by calling pd.read_csv() function.

2.	The Data Preprocessing
     
Model cannot take the image data directly so we need to perform some basic operations and process the data to make it ready for our neural network. 
The dimension of the training data is (60000*28*28). One more dimension is needed for the CNN model so we reshape the matrix to shape (60000*28*28*1). 
Normalized the entire dataset to proceed further to create the model.

3.	Create the model

Its time for the creation of the CNN model for this Python-based data science project. A convolutional layer and pooling layers are the two wheels of a CNN model.
 The reason behind the success of CNN for image classification problems is its feasibility with grid structured data.
 I develop the simple fully connected layers based on CNN and some additional features like Softmax and Relu functions. 
I flattened the matrix into vector and feed it into a fully connected layer like a neural network. I used adam optimizer for the model compilation.

4.	Train the model

To start the training of the model we can simply call the model.fit() function of Keras. It takes the training data, validation data, epochs, and batch size as the parameter.
The training of model takes some time. After successful model training, testing is done for a epoch of 10 times and later save the weights and model definition in the ‘model.h5’ file. 



5.	Evaluate the model

To evaluate how accurate our model works, we have around 10,000 images in our dataset. 
In the training of the data model, we do not include the testing data that’s why it is new data for our model. Around 98% accuracy is achieved.

6.	Create GUI to predict digits

To build an interactive window we have created a GUI. In this  you can draw digits on canvas, and by clicking a button, you can identify the digit. 
The Tkinter library is the part of Python standard library. Our predict() method takes the picture as input and then activates the trained model to predict the digit.
In GUI canvas you can draw a digit by capturing the mouse event and with a button click, we hit the predict_digit() function and show the results.



Conclusion:

This project is beginner-friendly and can be used by data science newbies.
 I have created and deployed a successful deep learning project of digit recognition with the most widely used algorithm CNN have been trained and tested on MNIST dataset.
 Utilizing these deep learning techniques, a high amount of accuracy is obtained. Using Kera’s as backend and TensorFlow as the software. 
I build the GUI for easy learning where we draw a digit on the canvas then we classify the digit and show the results.

Future Enhancements:

Further implementation is to make the model test with the robotics or use in creating artificial brain. As this model could be a very very small part of it.
 Digit recognition is an excellent prototype problem for learning about neural network and it gives a great way to develop more advanced techniques of deep learning.

 References:
	For dataset: https://kaggle.com
	For resolving errors: https://stackoverflow.com/
	For Understanding the topics :
	www.google.com
	www.youtube.com
	www.medium.com
	https://docs.python.org




 
