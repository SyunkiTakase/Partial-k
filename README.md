# Partial-k
ViTをPartial-kで学習するサンプルコード

![Partial-1](https://github.com/user-attachments/assets/3f7257af-b75d-44fa-bce2-0ea375a7eb3e)

## 動作環境
<details>
<summary>ライブラリのバージョン</summary>
 
* cuda 12.1
* python 3.6.9
* torch 1.8.1+cu111
* torchaudio  0.8.1
* torchinfo 1.5.4
* torchmetrics  0.8.2
* torchsummary  1.5.1
* torchvision 0.9.1+cu111
* timm  0.5.4
* tlt  0.1.0
* numpy  1.19.5
* Pillow  8.4.0
* scikit-image  0.17.2
* scikit-learn  0.24.2
* tqdm  4.64.0
* opencv-python  4.5.1.48
* opencv-python-headless  4.6.0.66
* scipy  1.5.4
* matplotlib  3.3.4
* mmcv  1.7.1
</details>

## ファイル＆フォルダ一覧

<details>
<summary>学習用コード等</summary>
 
|ファイル名|説明|
|----|----|
|vit_train.py|ViTを学習するコード(Fine-Tuning)．|
|vit_transfer.py|ViTを学習するコード(転移学習)．|
|vit_partial.py|ViTを学習するコード(Partial-k)．|
|trainer.py|学習ループのコード．|
</details>

## 実行手順

### 環境設定

[先述の環境](https://github.com/SyunkiTakase/ViT_Classification_Sample#%E5%8B%95%E4%BD%9C%E7%92%B0%E5%A2%83)を整えてください．

### 学習
ハイパーパラメータは適宜調整してください．

<details>
<summary>ViTの学習(CIFAR-10)</summary>

ViTのFine-Tuning 
```
python3 vit_train.py --epoch 1 --batch_size 128 --amp --dataset cifar10 --warmup_t 0 --warmup_lr_init 0
```
ViTの転移学習
```
python3 vit_transfer.py --epoch 1 --batch_size 128 --amp --dataset cifar10 --warmup_t 0 --warmup_lr_init 0
```
ViTのPartial-k(引数のlayerが学習する層になります．)
```
python3 vit_partial.py --epoch 1 --batch_size 128 --amp --dataset cifar10 --warmup_t 0 --warmup_lr_init 0 --layer 1
```
</details>

<details>
<summary>ViTの学習(CIFAR-100)</summary>

ViTのFine-Tuning 
```
python3 vit_train.py --epoch 1 --batch_size 128 --amp --dataset cifar100 --warmup_t 0 --warmup_lr_init 0
```
ViTの転移学習
```
python3 vit_transfer.py --epoch 1 --batch_size 128 --amp --dataset cifar100 --warmup_t 0 --warmup_lr_init 0
```
ViTのPartial-k(引数のlayerが学習する層になります．)
```
python3 vit_partial.py --epoch 1 --batch_size 128 --amp --dataset cifar100 --warmup_t 0 --warmup_lr_init 0 --layer 1
```
</details>

## 参考文献
* 参考にした論文
  * ViT
    * An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  * RandAugment
    * RandAugment: Practical automated data augmentation with a reduced search space
  * MixUp
    * mixup: Beyond Empirical Risk Minimization
  * CutMix
    * CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
  * Random Erasing
    * Random Erasing Data Augmentation

* 参考にしたリポジトリ 
  * timm
    * https://github.com/huggingface/pytorch-image-models
  * ViT
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
  * RandAugment
    * https://github.com/ildoonet/pytorch-randaugment
  * MixUp
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py
  * CutMix
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py
  * Random Erasing
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/random_erasing.py
