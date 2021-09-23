```{r setup, include=FALSE, echo=FALSE}
knitr::opts_knit$set(root.dir = getwd())
```

## CNN-based Reconstruction of Forces for Photoelastic Particles

### Abstract 

Photoelastic techniques have a long tradition in both qualitative and quantitative analysis of the stresses in granular materials. Over the last two decades, computational methods for reconstructing forces between particles from their photoelastic response have been developed by many different experimental teams. Unfortunately, all of these methods are computationally expensive. This  limits their use for processing extensive data sets that capture the time evolution of granular ensembles consisting of a large number of particles. In this repository, we present a novel approach to this problem which leverages the power of convolutional neural networks to recognize complex spatial patterns. The main drawback of using neural networks is that training them  usually requires a large labeled data set which is hard to obtain experimentally.  Hence, our networks were trained on synthetically generated data. In our paper, we show that a relatively small sample of real data is then needed to adapt models to perform well on the experimantal data. One could find more about how the models were constucted in our paper, currently available on [arXiv](https://arxiv.org/abs/2010.01163).

### Instructions on using the models

Running the model on the provided test data:

1. Install `Anaconda` from [here](https://docs.anaconda.com/anaconda/install/) with Tensorflow ([guide](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)). Additionally, you may need to install the following packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `scipy`. 
2. Clone our repository to your local machine. As part of our paper, we also provide best-performing trained models, which can be downloaded [here](). 
3. In your terminal application (inside the conda environment you created) type the following command: `python test.py user_input 0.008 2 6 <path to image_data folder> <path to models folder>`, where you substitute `<>` expressions with the appropriate paths.

Running the model on other data:

1. Apply steps `1-2` above.
2. In your terminal application (inside the conda environment you created) type the following command: `python test.py user_input [radius] [min number of forces] [max number of forces] <path to data folder> <path to models folder>`, where you substitute `<>` expressions with the appropriate paths and `[]` with the appropriate numbers based on your data (note: our model currently only supports `radius = 0.008`, `min number of forces > 1`, `max number of forces < 7`).