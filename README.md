# RetinexNet_Pytorch (Unofficial PyTorch Code)
A Pytorch implementation of RetinexNet

# Deep Retinex Decomposition for Low-Light Enhancement, BMVC'18 
Unofficial PyTorch code for the paper - Deep Retinex Decomposition for Low-Light Enhancement, BMVC'18 (Oral) 

Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu

The offical Tensorflow code is available [here](https://github.com/weichen582/RetinexNet). 

Please ensure that you cite the paper if you use this code:
```
@inproceedings{Chen2018Retinex,
 title={Deep Retinex Decomposition for Low-Light Enhancement},
 author={Chen Wei, Wenjing Wang, Wenhan Yang, Jiaying Liu},
 booktitle={British Machine Vision Conference},
 year={2018},
 organization={British Machine Vision Association}
}
```
### Training
Please download the training and testing datasets from [here](https://daooshee.github.io/BMVC2018website/). 
```
Data folder should like:
-- data_name(Ex. LOL)
  -- train
    -- low
    -- high
 -- test
    -- low
    -- high
```


And just run 
```
$ python train.py \
```


### Testing
For sample testing/prediction, you can run-
```
$ python predict.py
```
There is a pre-trained checkpoint available in the repo. You may use it for sample testing or create your own after training as needed. The results are generated (by default) for the data present in `./data/test/low/` folder, and the results are saved (by default) in `./results/test/low/` folder. 

##### This code is inspired by [RetinexNet_PyTorch from aasharma90](https://github.com/aasharma90/RetinexNet_PyTorch)
