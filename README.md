# DHDN
Pytorch Implementation of "Densely Connected Hierarchical Network for Image Denoising"

Second place winner of sRGB track and Third place winner of Raw-RGB track on [NTIRE 2019 Challenge on Real Image Denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Abdelhamed_NTIRE_2019_Challenge_on_Real_Image_Denoising_Methods_and_Results_CVPRW_2019_paper.pdf)

If you find our project useful in your research, please consider citing:

```
@inproceedings{park2019densely,
  title={Densely connected hierarchical network for image denoising},
  author={Park, Bumjun and Yu, Songhyun and Jeong, Jechang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
```

# Environment
python 3.6    
pytorch 1.0.0    
MATLAB

# Data
We used [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) dataset for training.[download](https://data.vision.ee.ethz.ch/cvl/DIV2K/)    
To generate training patches, use the matlab codes in ./data    
We used Kodak and BSD datasets for test datset. These datasets also needs to be modified by matlab codes in ./data to use our test codes.

# Training
As an example, use the following command to use our training codes
```
python main_color.py --batchSize 16 --lr 1e-4 --step 3 --cuda True --gpu 0,1 --checkpoint ./checkpoint
```

# Test
As an example, use the following command to use our test codes
```
python test_color.py --cuda True --model1 ./trained.pth --data ./data/noisy.mat --gt ./data/gt.mat --gpu 0 --result ./result/
```
To use our pretrained model, please download [here]()    
Test results are also available in ./data/results

# Results
We retrained our network as we found some problems of our paper version trained parameters.    
So the result of the pretrained models is a bit different from the paper.    
Color:
kodak| | | | | | |bsd| | | | | | 
----|----|----|----|----|----|----|----|----|----|----|----|----|----
noise level|10| |30| |50| |10| |30| |50| 
 |PSNR|SSIM|PSNR|SSIM|PSNR|SSIM|PSNR|SSIM|PSNR|SSIM|PSNR|SSIM
 noise|28.24|0.8127|18.93|0.3883|14.87|0.2195|28.30|0.8368|19.03|0.4385|15.00|0.2603

