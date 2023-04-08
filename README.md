![Alt Text](http://spoter.signlanguagerecognition.com/img/GitHub_banner.png)

> by **[Matyáš Boháček](https://github.com/matyasbohacek)** and **[Marek Hrúz](https://github.com/mhruz)**, University of West Bohemia <br>
> Should you have any questions or inquiries, feel free to contact us [here](mailto:matyas.bohacek@matsworld.io).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sign-pose-based-transformer-for-word-level/sign-language-recognition-on-lsa64)](https://paperswithcode.com/sota/sign-language-recognition-on-lsa64?p=sign-pose-based-transformer-for-word-level)

Repository accompanying the [Sign Pose-based Transformer for Word-level Sign Language Recognition](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html) paper, where we present a novel architecture for word-level sign language recognition based on the Transformer model. We designed our solution with low computational cost in mind, since we see egreat potential in the usage of such recognition system on hand-held devices. We introduce multiple original augmentation techniques tailored for the task of sign language recognition and propose a unique normalization scheme based on sign language linguistics.

![Alt Text](http://spoter.signlanguagerecognition.com/img/architecture_github.gif)

## Get Started

First, make sure to install all necessary dependencies using:

```shell
pip install -r requirements.txt
```

Reinstall torch v1.8.1 and torchvision that support CUDA. On Windows 10, NVIDIA GeForce 1080 Ti, with CUDA 11.6.
```commandline
pip install --upgrade --force-reinstall torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```



Place the data into the data folder.

To train the model on the WLASL100 dataset:
```commandline
python -m train  --training_set_path .\data\WLASL100_train_25fps.csv --validation_set_path .\data\WLASL100_val_25fps.csv --testing_set_path .\data\WLASL100_test_25fps.csv --experiment_name wlasl100 --num_classes 100      
```

For example, to train the model on the WLASL2000 dataset, on a Mac (the paths may be different and are platform-dependent):
```commandline
python -m train  --training_set_path ./data/WLASL2000_train.csv --validation_set_path ./data/WLASL2000_val.csv --testing_set_path ./data/WLASL2000_test.csv --experiment_name wlasl2000 --num_classes 2000      
```


In general, to train the model, simply specify the hyperparameters and run the following:

```
python -m train
  --experiment_name [str; name of the experiment to name the output logs and plots]
  
  --epochs [int; number of epochs]
  --lr [float; learning rate]
  
  --training_set_path [str; path to the csv file with training set's skeletal data]
  --validation_set_path [str; path to the csv file with validation set's skeletal data]
  --testing_set_path [str; path to the csv file with testing set's skeletal data]
```

If either the validation or testing sets' paths are left empty, these corresponding metrics will not be calculated. We also provide out-of-the box parameter to split the validation set as a desired split of the training set while preserving the label distribution for datasets without author-specified splits. These and many other specific hyperparameters with their descriptions can be found in the [train.py](https://github.com/matyasbohacek/spoter/blob/main/train.py) file. All of them are provided a default value we found to be working well in our experiments.

## Data

As SPOTER works on top of sequences of signers' skeletal data extracted from videos, we wanted to eliminate the computational demands of such annotation for each training run by pre-collecting this. For this reason and reproducibility, we are open-sourcing this data for WLASL100 and LSA64 datasets along with the repository. You can find the data [here](https://github.com/matyasbohacek/spoter/releases/tag/supplementary-data).

![Alt Text](http://spoter.signlanguagerecognition.com/img/datasets_overview.gif)

## License

The **code** is published under the [Apache License 2.0](https://github.com/matyasbohacek/spoter/blob/main/LICENSE) which allows for both academic and commercial use if  relevant License and copyright notice is included, our work is cited and all changes are stated.

The accompanying skeletal data of the [WLASL](https://arxiv.org/pdf/1910.11006.pdf) and [LSA64](https://core.ac.uk/download/pdf/76495887.pdf) datasets used for experiments are, however, shared under the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license allowing only for non-commercial usage.

## Citation

If you find our work relevant, build upon it or compare your approaches with it, please cite our work as stated below:

```
@InProceedings{Bohacek_2022_WACV,
    author    = {Boh\'a\v{c}ek, Maty\'a\v{s} and Hr\'uz, Marek},
    title     = {Sign Pose-Based Transformer for Word-Level Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {182-191}
}
```
