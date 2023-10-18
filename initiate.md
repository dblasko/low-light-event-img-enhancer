# Assignment 1 - Initiate

## Table of contents
- [Assignment 1 - Initiate](#assignment-1---initiate)
  - [Table of contents](#table-of-contents)
  - [The project: choice of topic, project type and motivation](#the-project-choice-of-topic-project-type-and-motivation)
  - [Related work that has been considered for my decisions](#related-work-that-has-been-considered-for-my-decisions)
  - [Intended approach](#intended-approach)
    - [Datasets](#datasets)
    - [Models](#models)
    - [Evaluation](#evaluation)
    - [How the project will be productized](#how-the-project-will-be-productized)
  - [Previsional work breakdown](#previsional-work-breakdown)
___

## The project: choice of topic, project type and motivation

The purpose of the topic I have chosen is to create a deep-learning-based low-light image enhancer specialized for event photographers. Its purpose is to help them restore dark photographs that would be deemed as unusable, and thus lead to the deletion of a souvenir, but also to save such dark pictures taken on a smartphone by anyone during a concert of a party. This project is a mix of mainly the "*bring your own data*" category, and to a certain extent, the "*bring your own methods* one, as explained in the next sections.

Low-light image-enhancing models usually focus on urban, home or landscape photographs. Due to the low presence of event-photograph in the typical training datasets, they often create abnormal artifacts when used on such pictures where humans are present and the lighting conditions are very specific. As a long-time hobby-photographer that also occasionally has party/concert/ceremony gigs, saving such photos through editing is time-consuming for photographers that then often simply disregard the picture as they have many others (that do not capture the same instant), and for the average person, it might not even be an option. This is what motivates this project where there is an opportunity to create value for a large group of professionals and anyone that has ever taken a dark picture on their phone and felt bad about it the day after.


## Related work that has been considered for my decisions

When considering my topic, and the approach I will be discussing in the later sections, these are the publications that I have considered and that have influenced my decisions:

- [1] [Jiawei Guo, Jieming Ma, Ángel F. García-Fernández, Yungang Zhang, Haining Liang,
**A survey on image enhancement for Low-light images**, 
2023.](https://doi.org/10.1016/j.heliyon.2023.e14558)
- [2] [Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao, **Learning Enriched Features for Real Image Restoration and Enhancement**, 2020.](https://doi.org/10.48550/arXiv.2003.06792)
- [3] [Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, **Image-to-Image Translation with Conditional Adversarial Networks**, 2017.](https://arxiv.org/abs/1611.07004)
- [4] [Chen Chen, Qifeng Chen, Jia Xu, Vladlen Koltun, **Learning to See in the Dark**, 2018.](
https://doi.org/10.48550/arXiv.1805.01934)


## Intended approach 

Some decisions made in this approach are linked to the my personal preferences on what I would like to work on out of technical interest: ideally, I would like to implement the model architecture in PyTorch myself, then pre-train it on an existing dataset to build up the low-level feature extractors (*because my own dataset cannot be too large*), and then gather a dataset specifically linked to my use-case semi-automatically and fine-tune my model on it. I would then like to evaluate both the pre-trained and fine-tuned models on the same data to observe the impact of my fine-tuning on this specific use-case. 

### Datasets

For the two-stage training approach described above, two datasets will be gathered:
- **Pre-training dataset**: for the larger pre-training dataset, publicly available low-light image-enhancement benchmarking datasets will be used. The goal is to gather about **2000 training, 100 validation and 100 evaluation images** by combining [LoL](https://paperswithcode.com/dataset/lol) (500 images), a subset of [ExDark](https://paperswithcode.com/dataset/exdark) (7368 images) and a subset of [Night2Day](https://huggingface.co/datasets/huggan/night2day), or only using some data from [Night2Day]() (20k images) depending on its quality that will be manually observed beforehand. 
  
- **Fine-tuning dataset**: this dataset would be specialized on the event-photography task. I would gather well-lit / edited event (concert, party, ceremony...) photographs from a mix of sources: my own event photographs that have undergone editing and selection to ensure that only good quality pictures would be delivered to clients, but also event images of satisfactory quality scraped from unrelated datasets and the internet (some free images from sites like [Pexels](https://www.pexels.com/search/dance_party/), specific classes from datasets like [Wider](https://paperswithcode.com/dataset/wider) that contains 61 event categories, similarly for [EiMM](http://loki.disi.unitn.it/~used/). However, for those, manual verification would be needed which would require some time. The priority for this fine-tuning data would be put on the quality rather than on the quantity). The goal would be to gather about **200 training, 20 validation and 30 evaluation images**. The `(obscure, lit)` pairs would be generated by programmatically darkening the images. In order to not only train a model that reverses a given darkening process, and also to compensate for the small size of the dataset, the images will be randomly augmented in that transformation: the blacks, shadows and brighteness will be darkened to a random level independently to simulate different lack-of-lighting situations. Data augmentation steps like random crops will also be used to make more training data of these images. While I would ideally like to publish the dataset, for example to HuggingFace, the methods used for gathering the data might be an obstacle for that. 

### Models

There are two achitectures I am considering for the project: a GAN-based pix2pix ([3]) or a CNN-based MirNet ([2]). Both have their advantages and limitations I will present below. My plan would be to implement the chosen architecture (without feedback, my choice would be the MirNet, and I would switch to pix2pix if I encounter a problem) from scratch in PyTorch, and then pre-train and fine-tune it. However, if this does not correspond to the expectations of the class (I would appreciate any feedback on this!), I would also be open to implementing and training both of them, and then comparing their performance on the same data (In that case, would it be acceptable to use a pre-implemented version, for example available in the `restorers` library?). 

- **Pix2pix**: generic image-to-image GAN architecture. It was initially conceived from image reconstruction from edges. The advantage of this approach is that it is a well documented approach. However, it is not specialized for this task, which leads to limitations such as producing unrealistic colors or blurry images, and comes with the potential training difficulties of GANs (mode collapse...) while being more compute-intensive.

- **MirNet**: CNN-based "niche" architecture designed specifically for such a task. I plan to use a charbonnier loss for its training. Its advantage is that it should lead to better results, has a more "traditional" training process, but there are very few technical resources or implementations available for it other than the publication and code of the original authors, which might lead to hidden time-costs during implementation.

Depending on the resulting workload, if possible, I would also like to do slight alterations to the architecture (different upsampling techniques, regularization) to observe their impact on the results, but this is kept optional depending on how the training goes, and if I train one or two models. In any case, model training and experiment tracking will be done with Weights & Biases.


### Evaluation

The resulting model(s) will be evaluated based on Peak Signal to Noise Ratio (PSNR) on the evaluation subset of both datasets after each stage, and the pre-trained model will also be evaluated on the fine-tuning test data to compare its performance with the fine-tuned model. 

### How the project will be productized 

The resulting model(s) will be made available through an endpoint in a Flask API that will be contained in a Docker container. 
A web-based client will be developed to allow users to upload their images and then view and download the enhanced versions.


## Previsional work breakdown 

- **Assembling the pre-training dataset**: 3 hours
- **Assembling the fine-tuning dataset**: 10 hours
- **Implementing the network architecture and debugging/testing it**: 10 hours
- **Implementing and debugging the training process** (*data loaders, pre-processing, data augmentation, training, fine-tuning and evaluation loops*): 12 hours
- **Running the training/fine-tuning experiments** (*includes parameter tuning, incrementing image/dataset/layer sizes...*): 17 hours
- **Implementing tests, documenting the code, CI pipeline** (*should be done in parallel throughout the other steps*): 7 hours
- **Wrapping the model in a Flask API and dockerizing it**: 4 hours
- **Creating a web-based client to interact with the API**: 3 hours
- **Writing the final report**: 6 hours
- **Preparing and filming the final presentation**: 6 hours
