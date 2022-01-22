# Watermark-Removal

![version](https://img.shields.io/badge/version-v1.0.0-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/tensorflow-v1.15.0-green.svg?style=plastic)
![license](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

An open source project that uses a machine learning based image inpainting methodology to remove watermark from images which is totally indistinguishable from the ground truth version of the image.

This project was inspired by the [Contextual Attention](https://arxiv.org/abs/1801.07892) (CVPR 2018) and [Gated Convolution](https://arxiv.org/abs/1806.03589) (ICCV 2019 Oral).

And also a shoutout to [Chu-Tak Li](https://chutakcode.wixsite.com/website) for his [Medium article series](https://towardsdatascience.com/10-papers-you-must-read-for-deep-image-inpainting-2e41c589ced0) that really gave me a deep insight into the image inpainting papers stated above

<img src="https://user-images.githubusercontent.com/51057490/140277713-c7d6e2b9-db62-4793-823a-25ed0c4e2771.png" width="45%"/> <img src="https://user-images.githubusercontent.com/51057490/140277781-5b5218bb-9044-4ec9-a349-eea93bc56d4a.png" width="45%"/> <img src="https://user-images.githubusercontent.com/51057490/140277929-3f187647-0e63-4bcb-b9f1-472f7558aae5.jpeg" width="45%"/> <img src="https://user-images.githubusercontent.com/51057490/140277957-6ddb7dec-25c8-42f1-8e39-be491d4f2248.png" width="45%"/> <img src="https://user-images.githubusercontent.com/51057490/140277983-265a1c9e-6093-4154-8252-838baca21c41.jpeg" width="45%" /> <img src="https://user-images.githubusercontent.com/51057490/140278002-56c4ae3d-6bfb-4ba3-aa02-7bd28474bfdf.png" width="45%" /> <img src="https://user-images.githubusercontent.com/51057490/140278030-d2a962ce-3722-43f1-b1bd-0ffde2aa7026.jpeg" width="45%" /> <img src="https://user-images.githubusercontent.com/51057490/140278040-10e401d7-4b7d-4d81-91fe-e9f01ef4ce7f.png" width="45%" /> <img src="https://user-images.githubusercontent.com/51057490/140278017-34862de0-86eb-40f0-b04b-7dc02fe38a77.jpeg" width="45%" /> <img src="https://user-images.githubusercontent.com/51057490/140278011-e0ae9ed0-e4ed-44ed-a9ac-28eb8456797a.png" width="45%" />

## Run

- use [Google colab](https://research.google.com/colaboratory/)

- First of all, clone this repo

      !git clone https://github.com/zuruoke/watermark-removal

- Change Directory to the repo

      !cd watermark-removal

- Since Google Colab uses the latest Tensorflow 2x version and this project uses 1.15.0, downgrade to Tensorflow 1.15.0 version and restart the runtime, (`although the new version of Google Colab does not need you to restart the runtime`).

      !pip install tensorflow==1.15.0

- Install tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym).

      !pip install git+https://github.com/JiahuiYu/neuralgym

- Download the model dirs using this [link](https://drive.google.com/drive/folders/1xRV4EdjJuAfsX9pQme6XeoFznKXG0ptJ?usp=sharing) and put it under `model/` (rename `checkpoint.txt` to `checkpoint` because sometimes google drive automatically adds .txt after download)

And you're all Set!!

- Now remove the watermark on the image by runing the `main.py` file

      !python main.py --image path-to-input-image --output path-to-output-image --checkpoint_dir model/ --watermark_type istock

## Citing

```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```
