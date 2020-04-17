# COGS 444 Honours Research Project for McGill University: Machine Learning for Brain Histological Image Analysis
The purpose of this research project is to design and implement software that uses machine learning to perform analysis on IHC images. Multiple machine learning models are evaluated to determine which has the highest performance on the Iba1 sample images provided by Stephanie Tullo. The decision tree model showed the strongest performance in both accuracy (77%) and recall (87%) and would be the most useful in laboratory settings. See the report for more in-depth results.

In this notebook, Iba1 Images go through a preprocessing pipeline where the signal to noise ratio is increased; the resulting image undergoes intensity thresholding, and candidate patches are extracted. Features including shape, texture, and histogram of oriented gradients are extracted from the patches. Shape features include solidity, orientation, diameter, area, eccentricity, convex area, major axis length, minor axis length, and extent. Texture features are based on the MR8 filter banks which include 36 bar and edge filters, a Gaussian filter, and a Laplacian of Gaussian filter. The eight highest responses are extracted to maintain rotation invariance. Histogram of Oriented Gradients (HoG) features were also extracted. These features are normalized using a min-max scaler and input to the models.

Eight models were trained and tested using Scikit-learn. The models are evaluated based on accuracy, recall, and precision as compared to manual analysis as the ground truth. Cross validation search was used to tune the hyperparameters of each model. For models with less than 1000 hyperparameter combinations, exhaustive grid search was performed. For models with more than 1000 hyperparameter combinations, randomized search was performed with 1000 iterations to limit computing time.

Model | Accuracy % | Recall % | Precision %
------------ | ------------- | ------------ | -------------
KNN | 70.42 | 78.79 | 78.79
Linear SVM | 71.83 | 73.74 | 83.91
RBF SVM | 71.13 | 74.75 | 82.22
Gaussian Process | 71.13 | 74.75 | 82.22
Decision Tree | 76.76 | 86.87 | 81.13
Random Forest | 71.13 | 74.75 | 82.22
MLP (Neural Network) | 70.42 | 74.75 | 81.32
AdaBoost | 71.83 | 75.76 | 82.42

The decision tree model is packaged as a python script to which researchers can provide their Iba1 IHC images as input, and the program reports the positive cell count in an accurate and precise manner. This code is be available on in another [repository](https://github.com/amyhynes/HistologyCellCounter) with instructions on how to install the requirements and run the script.

To get started running the code in this notebook, make sure you are using Python 3 and run the following in the command line: 'pip3 install -r requirements.txt'
