# Handwritten-Digit-Recognition-on-Android-Mobile

This project is based on recognition of single hamdwritten digit 0 to 9 using android app created in Android-Studio.

Here is several steps to run this whole project.

## 1. Install dependancies.

[**Install tensorflow 1.2.1**](https://www.tensorflow.org/versions/r1.2/install/install_linux#InstallingAnaconda) (also need to **install anaconda for python 2.7**)

Install other dependancies:
* install csv (pip install csv)
* install opencv (pip install opencv-python)
* install numpy (pip install numpy)

## 2. Download dataset from Site.

[**Donwload dataset for Training images**](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)

[**Download dataset for Training Labels**](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)

[**Download dataset for Test Images**](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)

[**Download dataset for Test Labels**](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

## 3. Run python files for training.

First, Run python file.
```
python mnist2jpg.py
```

Then run another python file.
```
python train_mnist.py
```
## 4. Build android project and run app on any android device(>lollipop)

First install android studio and config sdk with software as per software suggest.
After compliting all configuration,import project from **tfmobile**.
Import **output_graph.pb** and **output_labels.txt** which have created in **out/** when one run **python train_mnist.py**  in **tfmobile/asserts**.
Change in **ClassificationActivity.java** which is in **tfmobile/src/org/tensorflow/demo**. Changes are given below

private static final int INPUT_SIZE = 28;
private static final int IMAGE_MEAN = 128;
private static final float IMAGE_STD = 128.0f;
private static final String INPUT_NAME = "input";
private static final String OUTPUT_NAME = "output";

Then Build and Run **OR** check **tfmobile/gradleBuild/outputs/apk/debug** and get **tfmobile-debug.apk** to run on any adroid devices(>lollipop).

If is there any issue in making app one can take reference from [**application reference**](https://github.com/googlecodelabs/tensorflow-for-poets-2/tree/master/android/tfmobile).
