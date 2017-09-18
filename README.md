# xRay-Femur Image Segmentation Software 

Software aimed to perform segmentation of xRay femur images. It uses a Deep ConvNet as a predictor.
**Model assumptions**: 
- The input image must be in NifTi format. 
- The input image should be a x-Ray image (TAC images can fail). 
- The femur's head must be looking to the left, and to be in the top part of the image. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. This guide is designed for Linux OS and Python3.

### Prerequisites

Before using the software we need to install some dependencies. First, lets install the **Python3** ones. 

```
pip3 install --upgrade numpy nibabel scipy matplotlib 
```

The software needs TensorFlow 1.3+, you can install it by running the next line. 
```
sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp36-cp36m-linux_x86_64.whl 
```
In case that it does not work, follow this [guide](https://www.tensorflow.org/install/).

Finally, we need to install ANTs. You can find an installation guide [here](http://stnava.github.io/ANTs/). Once you've installed ATNs, move the binary file **ResampleImage** to /usr/local/bin/ so **it can be called from the terminal.**  


## Installation

This software does not need installation, just download it from the github repository.
```
git clone https://www.github.com/gergf/femurSoftwareSegmentation
```


## Running the tests

At this point everything should be ready to work. Let's tun a test to be sure everything is ok. Run the next line: 
```
python3 perform_segmentation.py --input ./test/test_sample.nii.gz --output ./test/test_result -v True
```
This should create three new files inside the test folder. One should be /test/test.nii.gz, which represents the segmentation of the sample saved in NifTi format; the second one should be the preprocessed sample, with name preprocess_test_sample.nii.gz, stored in ./test/. Because we set the flag of verbose to True, we have another file, named ./test/visualizations.png, which is a auto-generated visualization where you can check if the model's segmentation makes sense. With the test sample, you should get the next visualization:

![Example of visualization](https://user-images.githubusercontent.com/14791980/30539572-6826580c-9c73-11e7-9a12-13041117e1f3.png)

## Authors

* **Germán A. García Ferrando** -  [github](https://github.com/gergf)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
 
