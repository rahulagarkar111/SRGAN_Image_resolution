# SRGAN_Image_resolution Project for DA-526 IIT_Guwahati
## Group No - 2
### Group Members:
### Rahul Agarkar - 234101041
### Keshav Gupta - 234101023
### Shardul Nalode - 234101049
### Vikas Khurendra - 234101055

**********************************    Information about dataset  *********************************************
Dataset from: http://press.liacs.nl/mirflickr/mirdownload.html

Read high resolution original images and save lower versions to be used for SRGAN.

Here, we are resizing them to 128x128 that will be  used as HR images and 
32x32 that will be used as LR images.
This will be done in preprocessing folder.
Download the dataset and set path as mentioned in lanzcos.py file.

***************************************************************************************************************

Steps to run the project:
1)Download the files and dataset.
2)Install all dependencies mentioned in requirement.txt.
3)Use command streamlit run-app.py
4)Upload the image and you are good to go.

For training:
1)Get into training.ipnyb.
2)Upload the data(the preprocessing will create two folders named HR_images and LR_images.
3)set path wherever needed.
4)set hyperparameters according to you like no of epochs
5)You will get your model i.e .h5 file for generator

Testing:
1)test.py will take images from test_dataset directory which you have to make and generate 3 files i.e psnr.csv ssim.csv and average scores.
