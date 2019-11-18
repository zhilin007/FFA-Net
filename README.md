### AAAI 2020 Accepted paper "FFA-Net: Feature Fusion Attention Network for Single Image Dehazing" official implementation.

---

by Xu Qin, Zhilin Wang et al.    Peking university and Beihang university.

####Citation

To be determined.

####Dependencies and Installation

* python3
* PyTorch>=1.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib

####Datasets Preparation

Dataset website:[RESIDE](https://sites.google.com/view/reside-dehaze-datasets/) ; Paper arXiv version:[[RESIDE: A Benchmark for Single Image Dehazing](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1712.04143.pdf&sa=D&sntz=1&usg=AFQjCNHzdt3kMDsvuJ7Ef6R4ev59OFeRYA)]

<details>
<summary> FILE STRUCTURE</summary>

```
    FFA-Net
    |-- README.md
    |-- net
    |-- data
        |-- RESIDEV0
            |-- ITS
                |-- hazy
                    |-- *.png
                |-- clear
                    |-- *.png
            |-- OTS 
                |-- hazy
                    |-- *.jpg
                |-- clear
                    |-- *.jpg
            |-- SOTS
                |-- indoor
                    |-- hazy
                        |-- *.png
                    |-- clear
                        |-- *.png
                |-- outdoor
                    |-- hazy
                        |-- *.jpg
                    |-- clear
                        |-- *.png
```
</details>



####Usage

##### Train

如果你想使用tensorboard或查看中间预测结果的话,把main.py的注释删除 --记得翻译成英语

*If you have more computing resources, expanding `bs`, `crop_size`, `gps`, `blocks` will lead to better results*

* train network on `ITS` dataset

`python main.py --net='ffa' --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset='its_train' --testset='its_test' --steps=500000 --eval_step=5000`
* train network on `OTS` ataset

`python main.py --net='ffa' --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset='ots_train' --testset='ots_test' --steps=500000 --eval_step=5000`


##### Test
*Put your images in `net/test_imgs/` and modify `test.py` to suit your needs*
 * `python test.py`

