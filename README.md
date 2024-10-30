# LMFusion
LMFusion is a multimodal image fusion method for smoke conditions. Its characteristics are as follows:
1 LMFusion is suitable for multimodal  fusion in smoke scenes.
2 Latent feature mapping can deal with intermediate feature representation well
3 Piecewise optimizer increases the efficiency of training.
4 Experiments in two application areas demonstrate the advanced performance of LMFusion.

# Abstract
Multimodal image fusion can realize the high-quality integration of different complementary images, which has become one of the important research directions in artificial intelligence, and has a wide impact in the fields of security monitoring, medical diagnosis, intelligent manufacturing, traffic management and agricultural monitoring. Due to natural environment or man-made factors, the presence of smoke in smart manufacturing plants or other scenarios is often inevitable. To solve the multimodal image fusion with smoke environment or scene, a unified smoke removal image fusion algorithm based on latent feature mapping is proposed. The proposed algorithm is trained in two stages and an autoencoder architecture is adopted. In our proposed algorithm, the model structure and training method of latent feature mapping network are emphatically studied. And a novel piecewise optimizer is proposed for the training of the latent feature mapping network, which further improves the training efficiency in the second stage. The experimental results demonstate that compared with the latest and classical image fusion methods, our algorithm has greater multimodal fusion advantage and enhancement capability in smoke scene. The code of the fusion algorithm is available at https://github.com/windrunners/LMFusion.

# The core method of picture presentation
![1](https://github.com/user-attachments/assets/9901e1bd-17da-419a-8a5b-24934ddcf877)

Figure.1 The whole approach and fusion structure

![2](https://github.com/user-attachments/assets/fd466d85-4fd6-4cd3-be2f-12186916c571)

Figure.2 Model structure and details of LMFusion



# data set in the second stage
## data set for training
In the second stage of training, the LMFusion model adopts training data set RESIDE-ITS and test data set RESIDE-SOTS-indoor.
Download dataset and place the file in the main folder, then you can train.

## data set for test
Put the smoky test data into the corresponding file.

# train
Run the "main.py" file.

# test
Run the "test_image.py" file.

# The model trained in the first stage under the "models" folder
download link: 
A. 百度网盘

https://pan.baidu.com/s/1Mpk_KB8mTMCIKCkAS_r-1A 

password: pzqa

B. Google drive

https://drive.google.com/file/d/1TDKOOybjtN7qyduFUwrHO7hu2Rh105nH/view?usp=sharing


# Citation
```
@article{xxxx,
    title={LMFusion: Latent Feature Mapping Fusion for Multimodal Images in Smoke Scene},
    author={Zhao, Yangyang and Zheng, Qingchun},
    journal={xxxxxx},
    volume={x},
    number={x},
    pages={xx--xx},
    year={xxxx},
    publisher={xxxx}
}
```
