```{r setup, include=FALSE, echo=FALSE}
knitr::opts_knit$set(root.dir = getwd())
```

## CNN-based Reconstruction of Forces for Photoelastic Particles

### Abstract 

Photoelastic techniques have a long tradition in both qualitative and quantitative analysis of the stresses in granular materials. Over the last two decades, computational methods for reconstructing forces between particles from their photoelastic response have been developed by many different experimental teams. Unfortunately, all of these methods are computationally expensive. This  limits their use for processing extensive data sets that capture the time evolution of granular ensembles consisting of a large number of particles. In this repository, we present a novel approach to this problem which leverages the power of convolutional neural networks to recognize complex spatial patterns. The main drawback of using neural networks is that training them  usually requires a large labeled data set which is hard to obtain experimentally.  Hence, our networks were trained on synthetically generated data. In our paper, we show that a relatively small sample of real data is then needed to adapt models to perform well on the experimantal data. One could find more about hwo the models were constucted in our paper here: ...

### Instructions on using the models

Running the model on the provided test data:

1. Install `Anaconda` from [here](https://docs.anaconda.com/anaconda/install/).
2. Install `Tensorflow` on top of `Anaconda` from [here](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/).
3. Make sure you have the folllowing packages installed: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
4. For convenience, please, install `Spyder` IDE for Python in `Anaconda`.
4. Clone our repository to your local machine.
5. Open and run the `test.py` file. 
6. The output should be ![image](img_true_vs_pred_particles.png)

Running the model on other data:

1. Apply steps `1-4` above.
2. Convert your image data into `numpy` array such that the data matrix has dimesions `n * h * w * ch`, where `n` is the number of images, `h, w` are height and width of the images, and `ch` is the number of channels. Our models only works on the greyscale data, hence, the paraemter `ch` will be assumed to be equal to `1` and can be omitted in the matrix.  
The most straightforward way for converting images is through a Python package `Pillow`, the guidelines for which can be found [here](https://pillow.readthedocs.io/en/stable/index.html).
3. Save the data in the `.npy` file format and store it in the `image_data` folder. Rename your data to `data.npy`.
4. Open and run the `test.py` file. 




