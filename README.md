# Task-1_Object_Detection

Object detection is hottest topic of the computer vision field. Object detection is breaking into a wide range of industries, with use cases ranging from personal safety to productivity in the workplace. Object detection and recognition is applied in many areas of computer vision, including image retrieval, security, surveillance, automated license plate recognition, optical character recognition, traffic control, medical field, agricultural field and many more.

Single Shot object detection or SSD takes one single shot to detect multiple objects within the image. As you can see in the above image we are detecting coffee, iPhone, notebook, laptop and glasses at the same time.

It composes of two parts

-->Extract feature maps
-->Apply convolution filter to detect objects

SSD is faster than R-CNN because in R-CNN we need two shots one for generating region proposals and one for detecting objects whereas in SSD It can be done in a single shot.

In this project I have used OpenCV with MobileNet-SSD Network for object detection as it is fast and requires less computation power.
