# BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks || [Arxiv paper]() || [Blog]()

## Introduction
![BayesCap teaser](./figs/BayesCap.gif)

**Abstract.** High-quality calibrated uncertainty estimates are crucial for numerous real-world applications, especially for deep learning-based deployed ML systems. While Bayesian deep learning techniques allow uncertainty estimation, training them with large-scale datasets is an expensive process that does not always yield models competitive with non-Bayesian counterparts. Moreover, many of the high-performing deep learning models that are already trained and deployed are non-Bayesian in nature and do not provide uncertainty estimates. 
To address these issues, we propose **BayesCap** that learns a Bayesian identity mapping for the frozen model, allowing uncertainty estimation. **BayesCap** is a memory-efficient method that can be trained on a small fraction of the original dataset, enhancing pretrained non-Bayesian computer vision models by providing calibrated uncertainty estimates for the predictions without (i) hampering the performance of the model and (ii) the need for expensive retraining the model from scratch. The proposed method is agnostic to various architectures and tasks. We show the efficacy of our method on a wide variety of tasks with a diverse set of architectures, including image super-resolution, deblurring, inpainting, and crucial application such as medical image translation. Moreover, 
we apply the derived uncertainty estimates to detect out-of-distribution samples in critical scenarios like depth estimation in autonomous driving.


***TLDR:*** This is the official [PyTorch](https://pytorch.org/) implementation of BayesCap (from *ECCV 2022*) that allows estimating calibrated uncertainty for pre-trained (frozen) computer vision regression models in fast and efficient manner.

The structure of the repository is as follows:
```
BayesCap
|-ckpt/ (has all the checkpoints)
|-src/ (has all the codes and notebooks)
|-demo_examples/ (has some example images)
```



