# Adam

In December 2014, two scholars, Kingma and Lei Ba, put forward the Adam optimizer combining the advantages of the two optimization algorithms, AdaGrad and RMSProp. 
The First Moment Estimation of the gradient and the SecondMoment Estimation are taken into comprehensive consideration to calculate the update step size. 

**It mainly contains the following significant advantages**

- 1.Simple implementation, high calculation efficiency, less memory demand.
- 2.Parameter update is not affected by the scaling transformation of gradient.
- 3.And usually don't need to adjust or only a few tweaks. 
- 4.The update step length can be limited within the scope of the general vector (initial). 
- 5.Can be naturally implementation step annealing process (vector automatically adjust). 
- 6.Very suitable for application to large-scale data and parameters of scene. 
- 7.Objective function is suitable for the unstable. 
- 8.Suitable for gradient sparse or there is a big noise gradient composite 

**Adam in many cases as the optimizer of default work performance is good.**

*Algorithm*

![Adam](https://upload-images.jianshu.io/upload_images/10046814-c2db68e06531e759.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/897/format/webp)

## Adam updates the rules

Calculate the gradient of t time step:

![](https://upload-images.jianshu.io/upload_images/10046814-cce3170765aea25d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/185/format/webp)

- First

The exponential moving average of the gradient is calculated, and m0 is initialized to 0.

Similar to Momentum algorithm, gradient Momentum of previous time step is comprehensively considered.

The 1 coefficient is the exponential decay rate and controls the weight distribution (momentum and current gradient), usually taking a value close to 1.

The default is 0.9

![](https://upload-images.jianshu.io/upload_images/10046814-0e52be5dc8a5c11a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/268/format/webp)

- Second

Calculate the exponential moving average of the gradient squared and initialize v0 to 0.

The 2 coefficient is the exponential decay rate, which controls the effect of the gradient squared before.

Similar to the RMSProp algorithm, the gradient square is weighted to the mean.

The default is 0.999

![](https://upload-images.jianshu.io/upload_images/10046814-a5338b191eae4d91.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/256/format/webp)

- Thirdly

As m0 is initialized to 0, mt tends to be 0, especially in the early stage of training.

Therefore, deviation correction of the gradient mean value mt is needed here to reduce the impact of deviation on the initial training stage.

![](https://upload-images.jianshu.io/upload_images/10046814-5874367967271fae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/201/format/webp)

- Fourthly

It is similar to m0, because the initialization of v0 to 0 leads to vt bias to 0 in the initial stage of training, which is corrected.

![](https://upload-images.jianshu.io/upload_images/10046814-b5773ebc387c35d9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/181/format/webp)

- Fifth

Update the parameter, the initial learning rate times the ratio of the gradient mean to the square root of the gradient variance.

Where the default learning rate =0.001

Epsilon = 10 ^ - 8, avoid divisor to 0.

It can be seen from the expression that the updated step size calculation can be adjusted adaptively from the Angle of gradient mean value and gradient square, rather than directly determined by the current gradient.

![](https://upload-images.jianshu.io/upload_images/10046814-59e992b67938aec9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/302/format/webp)


## Adam defects and improvements

Although Adam algorithm has become the mainstream optimization algorithm at present, the best results in many fields (such as object recognition in computer vision and machine translation in NLP) are still obtained by using SGD of Momentum. The results of Wilson et al. 's paper show that the adaptive learning rate method (including AdaGrad, AdaDelta, RMSProp, Adam, etc.) is generally worse than Momentum algorithm in terms of object recognition, character-level modeling, and grammatical component analysis. 

** Aiming at the problem of Adam and other adaptive learning rate methods, the improvement is mainly in two aspects**

- 1.Decoupling weight attenuation

Every time the gradient is updated, it is attenuated at the same time (the attenuation coefficient w is slightly less than 1) to avoid excessive parameters.

![](https://upload-images.jianshu.io/upload_images/10046814-d005ff2c3d4e3fa7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/134/format/webp)

In the process of Adam optimization, the parameter weight attenuation term is added. Decoupling learning rate and weight attenuation two super parameters, can debug and optimize two parameters alone.

![](https://upload-images.jianshu.io/upload_images/10046814-ee99659d3067ba4e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/315/format/webp)

2. Modified exponential moving mean

Several recent papers have shown that lower [if! For example, in the training process, a certain mini-batch contains gradient information with relatively large amount of information. However, due to the low frequency of such mini-batch, the exponential moving mean will weaken their role (because the weight of the current gradient and the square of the current gradient are both small), resulting in poor convergence in this scenario.

![](https://upload-images.jianshu.io/upload_images/10046814-2e0cedb309fef898.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/272/format/webp)

The author puts forward Adam's deformation algorithm AMSGrad.

AMSGrad USES the largest gradient to update the gradient, unlike Adam's algorithm, which USES historical moving averages. The author observed better effects than Adam on the small batch data set and cifar-10.