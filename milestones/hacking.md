# Assignment 2 - Hacking <!-- omit from toc -->

<!-- This is a Github specific syntax that might not render correctly if you view this file in a local editor. The intended rendering can be viewed at https://github.com/dblasko/low-light-event-img-enhancer/blob/main/milestones/hacking.md -->
> [!NOTE]
> This document is a complement to the `README.md` file at the root of the project to provide further details about the second assignment. General information about the structure of the project, the features and how to run different elements can be found in the `README.md` file and its content is part of the assignment 2 delivery.

## Table of contents <!-- omit from toc -->
- [Evolutions from the description in the proposal](#evolutions-from-the-description-in-the-proposal)
  - [The data used in the project](#the-data-used-in-the-project)
  - [The model](#the-model)
- [Model results](#model-results)
- [Features and elements that have been delivered, and the process and decisions related to developing them](#features-and-elements-that-have-been-delivered-and-the-process-and-decisions-related-to-developing-them)
- [Time log](#time-log)



___


## Evolutions from the description in the proposal
### The data used in the project
**Pretraining**: The pretraining experiments started using the Night2Day dataset, as planned in the proposal as it is very large. However, I realized while testing the model that the dataset was not appropriate for this task: the dark images are not necessarily low-light, they are rather night images (with appropriate exposure), leading to the model learning the transform, for example, night landscapes into their daily equivalent rather than learning to recover under-exposed picture which was the task I was aiming for. Therefore, after experimentation, I switched to the LoL dataset I also cited in the proposal as it is more appropriate for the task.

**Finetuning**: Initially, I planned to scrape data from the internet to build a dataset of low-light images of events. However, that approach would not allow me to publish the dataset, and presented potential copyright issues. Instead, I decided on putting more effort into assembling my own dataset of event phtography that I could then also publish. During my experimentation, I also tried adding a small proportion of my photographs that aren't directly event pictures but that are in the similar style, and that improved performance as well. 

**Data augmentation**: Some additional data augmentation was experimented with and added. When generating the datasets, I did not only darken the images but also experimented with implementing the addition of random Gaussian noise, with various parameters, to emulate the noise that is present in low-light images taken on cameras. This addition turned out improving the robustness of the model to noise, and it did improve performance. This step was done at data generation and not at training time due to the important additional training time it otherwise generated.

### The model

As planned in the proposal, I have used the MIRNet architecture, and have implemented it and its different building blocks from scratch in PyTorch. I have experimented with different sizes for the different elements and tweaked some aspects, and generally chose the different sizes through both experimentation and automatic hyperparameter tuning.  
The model has been pre-trained on public data and then fine-tuned on the dataset I created, and both models have been released and evaluated to make them comparable, as I had planned. 

## Model results
Two final models have been [released](https://github.com/dblasko/low-light-event-img-enhancer/releases):
- **the MIRNet pre-trained for 100 epochs on 64x64 images, with a batch size of 8 and learning rate of 1e-4**  
    <img src='https://lh3.googleusercontent.com/drive-viewer/AK7aPaB7w7a3mQO4sWoUMtIxKKOBhlfVrvdGaeDZiJU72KKpXKYCeigEHmRYCYMQbogjWGqifrPTJj26Yx0MFmtVSaTxc7VRqA=s1600' width='700'>
- **the same model fine-tuned for 100 further epochs on the created event-dataset with early-stopping kicking in at 35 epochs (not visible well on the plots below, but from there validation loss and PSNR only get worse)**.   
    <img src='https://lh3.googleusercontent.com/drive-viewer/AK7aPaD1UWQcq3_FBuxii0DVBQbo-OMVYoSLTv9Jrd7OZZMU75UCtvVjBS21mgk5cG9M_wTDG81nrcQbNPvCcfuRPA6Igc06=s1600' width='700'>

Both have been evaluated on the same unseen subset of the task-specific event dataset, and the results are presented below.

| Model       | PSNR (*higher is better*) |
| ----------- | :-----------------------: |
| Pre-trained |           18.57           |
| Fine-tuned  |           19.64           |

**The target value set for the PSNR was of 17**, and the pre-trained model already exceeded it. The fine-tuned model improved the PSNR by 1.07, which is good but not as much as I had expected (and many different versions of finetuning with different parameters have been tried).   
This is likely due to the fact that the pre-trained model was already very good at the task, and generalized better than I expected. The architecture, with this size, was sufficient to learn a version of low-light-image-enhancement from the pre-training dataset that generalizes well to unseen image styles. Fine-tuning, even for longer, or with more data (I had an additional dataset of 1500 wedding pictures, that I did not include because I was not allowed to publish those - using those for fine-tuning as well did not improve performance), plateaued at a PSNR of ~19.6 at best, and other hyperparameter combinations led to catastrophic forgetting very fast or overfitting. Different combinations of frozen layers also had to be tried to obtain this fine-tuned version that does not suffer from catastrophic forgetting.  
Thus, my intuition was that the model had already reached its maximal capacity, which led to experiments with a larger model (varied the numbers of different types of blocks as well as image resolution), but those did not generalize better and mostly tended towards learning noise despite regularization or not being better, and were also limited by the compute available to me to fully train very large models even on Colab Pro, leading to the conclusion that this was the best performance I could obtain here. This model had already resulted from a dimension augmentation (otherwise, fine-tuning was not beneficial). All these observations were also reflected in the fine-tuning process, where even for the best version, early stopping was key as after the optimum at ~35 epochs, the model started overfitting and the PSNR started decreasing again and the validation loss going up too.   
The task turned out to be not as domain-specific as e.g. object detection, and thus the impact of fine-tuning was limited.

<img src='https://lh3.googleusercontent.com/drive-viewer/AK7aPaAuiGdhBTHwaKjzAfeNEaFJM2dJL48w8Y9j6TS9_hvkBBLzENQQ5r6yUbqfu5_QMIY1ynDj0yDsjrZmnyGCVvMuez2HJg=s1600' width="500">  

*Fine-tuning validation PSNR curve*

However, this discussion of the quantitative results has to be nuanced through the qualitative evaluation. The fine-tuning does not only improve the PSNR, but most importantly it enhances the visual quality of the generated images, as can be seen below. Manual evaluation of the models has revealed that the fine-tuned version leads to predicted images with more accurate constrast and sharper edges, which were the main limitations of the pre-trained version when used on event photography. The pre-trained model has a tendency towards generating washed-out colors and lighter grays leading to lower contrasts. As the event photography is most often quite colorful, this was exacerbated on such pictures - and the fine-tuning helped with this aspect. The pre-trained model also has a tendency towards blurring the edges of objects, which is also improved by the fine-tuning to some extent. It is interesting to note that those improvements got stronger as more diverse data was introduced in the fine-tuning dataset, with a small proportion of images not being directly event pictures.  
To illustrate this, sample images for both models can be viewed under the respective releases of the [pre-trained model](https://github.com/dblasko/low-light-event-img-enhancer/releases/tag/mirnet-pretrained-LoL-1.0.0-100epochs) and the [fine-tuned model](https://github.com/dblasko/low-light-event-img-enhancer/releases/tag/mirnet-finetuned-1.0.0-100epochs).

## Features and elements that have been delivered, and the process and decisions related to developing them

- As part of the assignment, two models have been [publicly released](https://github.com/dblasko/low-light-event-img-enhancer/releases), one pre-trained and one fine-tuned on the task-specific dataset, as described in the previous sections.
- A fine-tuning dataset of low-light event photography has been curated and [released](https://github.com/dblasko/low-light-event-img-enhancer/releases), along with a [PyTorch dataloader to use it](https://github.com/dblasko/low-light-event-img-enhancer/blob/main/dataset_generation/PretrainingDataset.py).  
    Some decisions had to be made when preparing both the pre-training and fine-tuning datasets, and they have been described and discussed above in *[Evolutions from the description in the proposal](#evolutions-from-the-description-in-the-proposal)*.   
    Additionally, different decisions have been made, backed by experimentation, about the data augmentations and transformations used (random levels of darkening, addition of random gaussian noise, image resizing and padding to the model's expected aspect ratio, square cropping at training, mixup starting at later epochs...). Some data augmentations also had to be discarded, as for example random rotations which had no impact whatsoever as the model is invariant to rotations (that property is also used at inference to infer on both vertical and horizontal images without needing to crop them).
- The MIRNet architecture has been implemented from scratch in PyTorch in a modular way (*as can be observed in the [definitions](https://github.com/dblasko/low-light-event-img-enhancer/tree/main/model)*), along with utilities like charbonnier loss, some custom transformations like the gaussian noise and mixup.
- The model has been pre-trained and fine-tuned as previously described. Many experiments have been run with tweaks to hyperparameters, sizes of different model dimensions, data augmentation used, the data mix (*e.g. including a few more diverse photographs not related to events*) to observe their impact and learn from them for the final model training. 
- Hyperparameter tuning has been implemented and run. WANDB sweeps have been used for this, leading to a great overview of the impact of each hyperparameter change and of the general patterns, and keeping a trace of each experiment. Most sweeps have been run for fewer epochs due to compute limitations, and once the grid was narrowed down, another sweep with longer training has been run to ensure that the dynamics did not change when training further.
- The training and fine-tuning processes have been implemented in a single generic program that is configurable, reproducible and trackable. The [training script](https://github.com/dblasko/low-light-event-img-enhancer/blob/main/training/train.py) is used for both phases of training and relies on versionable configuration files to define all parameters of the training run (*refer to the documentation for more details*). WANDB is used to track all runs, their parameters and evolution of performance metrics, allowing for reproducible and comparable experiments. Early-stopping has been implemented as well as checkpointing that removes redundant checkpoints if the model improves for a few epochs (due to large model size, to spare storage) and that allows resuming the training at any point with the same configuration file (*as Google Colab can potential interrupt it at any time*).  All training runs have been organized, grouped and named consistently in WANDB to keep a good overview of the different experiments at all times.  
    ![Screenshot from the WANDB dashboard](https://lh3.googleusercontent.com/drive-viewer/AK7aPaAQuVt--8c09DvglghsHLPruTIvvUja3uhfRoeyHNj0BBkcOjBWXfIHvNdraCokXPvz3KeCt_CA4jqtpJQLtL2jEYoc-A=s1600)  
    *Screenshot of part of the WANDB dashboard including a subset of the runs*
- Tests have been implemented for the project. Both unit and integration tests are present in the [tests folder](https://github.com/dblasko/low-light-event-img-enhancer/tree/main/tests). Unit tests verify the correct dimensions of the output of each layer/block of the MIRNet implementation. The data pipeline is tested too (*data loading, augmentations, batching, normalization...*), as well as the loss function that has been implemented (*effect of its hyperparameter, verifying it is differentiable, expected behavior for an input close/far from the target*). Finally, the training/validation procedures have been tested too on a small subset of the data and for a few epochs (*adaptations to ensure this test can run on the CPU of the CI server*) (*the loss should decrease over multiple epochs, no weights becoming NaN or inf, the validation loop should not change the model's weights*...).
- A continuous-integration pipeline has been set up using Github Actions. At each commit on the main branch, the code is linted and all tests are run (even the model training tests that have been designed to be runnable on the CI server CPU). The status of the pipeline for each commit can be seen on [the Github repository](https://github.com/dblasko/low-light-event-img-enhancer), and the definition of the pipeline (Ubuntu-based container that installs the dependencies and runs the tests and linting) can be viewed [here](https://github.com/dblasko/low-light-event-img-enhancer/blob/main/.github/workflows/continuous-integration.yml).
- Scripts and functions for model inference have been implemented. There is a [generic inference function and script](https://github.com/dblasko/low-light-event-img-enhancer/blob/main/inference/enhance_image.py) that can be used for inference on any image of any size (*images are padded/rotated as needed for the inference to be run on any aspect ratio and size, in a transparent way to the user as the output is of same dimension as the input*), as well as a [script to evaluate the model on a batch of images in a single command](https://github.com/dblasko/low-light-event-img-enhancer/blob/main/inference/visualize_model_predictions.py) (*runs inference on the full dataset, measures PSNR and generates a visual grid of (dark image, light image, inferred image*) rows for all images of the dataset and saves it as a png file). This allows for easy qualitative and quantitative evaluation of different versions of the model at different stages, which helped a lot to make adjustments during the development and improvement process as well. 
- Finally, the entire project has been documented thoroughly through comments and the [README file](https://github.com/dblasko/low-light-event-img-enhancer/blob/main/README.md) that documents the project structure, use, tooling, releases, etc.



## Time log 
| Task                                                                                                                                            | Anticipated workload |       Time spent       |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | :------------------: | :--------------------: |
| Assembling the pre-training dataset                                                                                                             |       3 hours        |  3 hours </br> *(=)*   |
| Assembling the fine-tuning dataset                                                                                                              |       10 hours       |  7 hours </br> *(-3)*  |
| Implementing the network architecture and debugging/testing it                                                                                  |       10 hours       | 16 hours </br> *(+6)*  |
| Implementing and debugging the training process (*data loaders, pre-processing, data augmentation, training, fine-tuning and evaluation loops*) |       12 hours       | 20 hours </br> *(+8)*  |
| Running the training/fine-tuning experiments (*includes parameter tuning, incrementing image/dataset/layer sizes...*)                           |       17 hours       | 28 hours </br> *(+11)* |
| Implementing tests, documenting the code, CI pipeline, experiment tracking ... (*should be done in parallel throughout the other steps*)        |       7 hours        | 14 hours </br> *(+7)*  |
