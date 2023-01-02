# Drone and Bird Detection using YoloV5
   It will help to identify where the entity is Bird or Drone it is mostly usedful for the places where more tight security is needed or more on country Borders where the enemy try to spy on military bases.
	 
# Aim and Objectives
# Aim:
To create a real-time video Drone and Bird detection system which will detect objects based on whether it is Bird or Drone for Military Borders Confidental Process.

# Objectives:
➢The main objective of the project is to create a program which can be either run on any pc with YOLOv5 installed and start detecting using the camera module on the device.

➢ Using appropriate datasets for recognizing and interpreting data using machine learning.

➢ To show on the optical view finder of the camera module whether objects are Bird or Drone.

# Abstract:
➢ An object is classified based on whether it is Bird or Drone is detected by the live feed from the system’s camera.

➢ We have completed this project by using YoloV5 Algorthim.

➢ A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

➢ One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

➢Military informations are highly confidential and can't get leaked as it will help other countries to build conspiracy against one another and Drone is one of them which helps to get the visuals from above and start bargaing into country Boarders.

➢ Therefore, skillfully designed video based Drone and Bird detection systems, using surveillance cameras in open space environments, can be the key to providing early warning signals. These systems utilise videos obtained from surveillance cameras and subsequently process. them to detect either Drone or Bird is entering from the boarders.

# Introduction:
➢ This project is based on a Drone and Bird detection model with modifications. We are going to implement this project with Machine Learning.

➢ This project can also be used to gather information about what category of Drone and Bird does the object comes in.

➢ The objects can even be further classified into Bird or Drone based on the image annotation we give in roboflow.

➢ Drone and Bird detection sometimes becomes difficult as certain mixed together and gets harder for the model to detect. However, training in Roboflow has allowed us to crop images and also change the contrast of certain images to match the time of day for better recognition by the model.

➢ Neural networks and machine learning have been used for these tasks and have obtained good results.

➢ Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Drone and Bird detection as well.

➢Video-based Drone detection is currently a standard technology due to image processing, computer vision, and Artifcial Intelligence. These systems have remarkable potential advantages over traditional methods, such as a fast response and wide detection areas. 

➢UAVs can be hijacked or manipulated. They can also trespass into authorized areas such as airports and military zones. While convenient surveillance is an advantageous use of drones, it can become a disadvantage with severe consequences when done by third parties.

➢ With the advent of computer vision and image processing, vision based Drone detection techniques are widely used in recent times. This technique provides numerous advantages over the conventional system such as quicker response and wider coverage area. Many algorithms are available for Drone and Bird detection. These algorithms use convolution neural networks which give way better accuracy in detection than the conventional methods of detection.

➢Deep learning algorithms can learn the useful features for Bird and Drone detection from a video source. Convolutional neural networks are a branch of deep learning that can extract topological properties from an image.

➢ In our approach we use convolutional neural networks to train the system for intelligently detecting drone. This is done by training the system on a very diverse dataset of Drone,Bird  images. This can successfully improve the accuracy of detection. This method will improve the accuracy of detection than the existing vision based models.

# YoloV5:
➢ Identification of objects in an image considered a common assignment for the human brain, though not so trivial for a machine. Identification and localization of objects in photos is a computer vision task called ‘object detection’, and several algorithms has emerged in the past few years to tackle the problem. One of the most popular algorithms to date for real-time object detection is YOLO (You Only Look Once).
➢Setup:
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5

# Methodology:
The Drone and Bird detection system is a program that focuses on implementing real time Drone and Bird detection. It is a prototype of a new product that comprises of the main module: Drone and Bird detection and then showing on view finder whether the object is either Drone, Bird or not.

Droen and Bird Module

This Module is divided into two parts:

1] Drone and Bird detection

➢ Ability to detect the location of object in any input image or frame. The output is the bounding box coordinates on the detected object.

➢ For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

➢ This Datasets identifies object in a Bitmap graphic object and returns the bounding box image with annotation of object present in a given image.

2] Classification Detection

➢ Classification of the object based on whether it is Drone and bird.

➢ Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

➢ There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

➢ YOLOv5 was used to train and test our model for various classes like Drone and Bird. We trained it for 150 epochs and achieved an accuracy of approximately 97%.

# Setup:
nstallation
Initial Setup

Remove unwanted Applications.

sudo apt-get remove --purge libreoffice*

sudo apt-get remove --purge thunderbird*

Create Swap file

sudo fallocate -l 10.0G /swapfile1

sudo chmod 600 /swapfile1

sudo mkswap /swapfile1

sudo vim /etc/fstab###########add line###########

/swapfile1 swap swap defaults 0 0

Cuda Configuration

vim ~/.bashrc#############add line #############

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1source ~/.bashrc

Udpade and Upgrade a System

sudo apt-get update

sudo apt-get upgrade

Install Some Required Packages

sudo apt install curlcurl https://bootstrap.pypa.io/get-pip.py -o get-pip.pysudo python3 get-pip.pysudo apt-get install libopenblas-base libopenmpi-devvim ~/.bashrc

add line
export OPENBLAS_CORETYPE=ARMV8source ~/.bashrcsudo pip3 install pillow

Install Torch

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whlmv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m- linux_aarch64.whlsudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl#Check Torch, output should be “True”

sudo python3 -c "import torch; print(torch.cuda.is_available())"

Installation of Torchvision.

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision

cd torchvision/

sudo python3 setup.py install

Clone yolov5

cd

git clone https://github.com/ultralytics/yolov5.git

cd yolov5/sudo pip3 install numpy==1.19.4# comment torch,PyYAML and torchvision in requirement.txtsudo pip3 install --ignore-installed PyYAML>=5.3.1

sudo pip3 install -r requirements.txt

Download weights and Test Yolov5 Installation on USB webcam

sudo python3 detect.py

sudo python3 detect.py --weights yolov5s.pt --source 0

Drone and Bird Dataset Training

We used Google Colab And Roboflow

Train your model on colab and download the weights and past them into yolov5 folder link of project

Running Drone and Bird Detection Model

source ‘0’ for webcam and for video recognition just upload the video [source 'video.mp4'].

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

Output Video

# Demo:

https://user-images.githubusercontent.com/88026146/201495117-a8c5fd80-2090-4506-9200-d32c6aad1ea9.mp4


# Advantages:
➢ Video-based Drone detection is currently a standard technology due to image processing, computer vision, and Artificial Intelligence. These systems have remarkable potential advantages over traditional methods, such as a fast response and wide detection areas.

➢Deep learning techniques have the advantage of extracting the features automatically, making this process more effective and dramatically improving the state-of-the-art in Image Classification and object detection methods

➢ It can then convey to the person who present in control room if it needs to be completely automated

➢ When completely automated no user input is required and therefore works with absolute efficiency and speed.

➢ It can work around the clock and therefore becomes more cost efficient.
  
# Application:
➢Detects object class like Drone or Bird in a given image frame or view finder using a camera module.

➢ Can be used in various places.

➢ Can be used as a refrence for other ai models based on Drone and Bird detection.

# Future Scope:
➢ As we know technology is marching towards automation, so this project is one of the step towards automation.

➢ Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

➢ Drone and Bird segregation will become a necessity in the future due to rise in spying from another countries which is not benefically good.

➢ In future our model which can be trained and modified with just the addition of images can be very useful.

# Conclusion:
➢ In this project our model is trying to detect objects and then showing it on view finder, live as what their class is as whether they are Drone and Bird as we have specified in Roboflow.

➢Considering the fair Drone detection accuracy of the CNN model, it can be of assistance to military teams by alerting on time, thus preventing huge losses.

Refrences
1] Roboflow :- https://roboflow.com/

2] Datasets or images used: https://www.gettyimages.in/search/2/image?license=rf&alloweduse=availableforalluses&family=creative&mediatype=photography&phrase=drone&sort=mostpopular

3] Google images

References:

1. Zeng, Y.; Zhang, R.; Lim, T.J. Wireless Communications with Unmanned Aerial Vehicles: Opportunities and Challenges. IEEE Commun. Mag. 2016, 54, 36–42.
2. World Economic Forum. Drones and Tomorrow’s Airspace. 2020. Available online: https://www.weforum.org/communities/drones-and-tomorrow-s-airspace (accessed on 13 January 2022).
3. Scott, G.; Smith, T. Disruptive Technology: What Is Disruptive Technology? Investopedia 2020. Available online: https://www.investopedia.com/terms/d/disruptive-technology.asp/ (accessed on 13 January 2022).
4. Germen, M. Alternative cityscape visualisation: Drone shooting as a new dimension in urban photography. In Proceedings of the Electronic Visualisation and the Arts (EVA), London, UK, 12–14 July 2016; pp. 150–157.
5. Kaufmann, E.; Gehrig, M.; Foehn, P.; Ranftl, R.; Dosovitskiy, A.; Koltun, V.; Scaramuzza, D. Beauty and the beast: Optimal methods meet learning for drone racing. In Proceedings of the IEEE International Conference on Robotics and Automation, Montreal, QC, Canada, 20–24 May 2019; pp. 690–696.
6. Kaufmann, E.; Loquercio, A.; Ranftl, R.; Dosovitskiy, A.; Koltun, V.; Scaramuzza, D. Deep drone racing: Learning agile flight in dynamic environments. In Proceedings of the Conference on Robot Learning (CoRL), Zürich, Switzerland, 29–31 October 2018; pp. 133–145.
7. Ahmad, A.; Cheema, A.A.; Finlay, D. A survey of radio propagation channel modelling for low altitude flying base stations. Comput. Netw. 2020, 171, 107122.
