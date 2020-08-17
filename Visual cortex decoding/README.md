# Visual cortex decoding (mind reading)

Acquiring BOLD fMRI images from the brain of a participant who is viewing images or watching a movie sets up an interesting machine learning problem.

Given the brain activity (BOLD fMRI signal in visual cortex) and a set of images displayed to the subject, train a model that can reconstruct what a person is seeing for some unknown images, based on the BOLD signal in their visual cortex.

[Article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) describing dataset, read materials and methods to understand dataset.

Dataset: [Deep Image Reconstruction](https://openneuro.org/datasets/ds001506/versions/1.3.1) (from Openneuro)

Use only a single subject to reduce the data size (also can extend it to all subjects). In the natural image presentation experiments, the stimulus images are named as 'n03626115_19498', where 'n03626115' is ImageNet/WorNet ID for a synset (category) and '19498' is image ID. Download the geometric shape images and alphabetical shape images at openneuro.

All information needed to extract events is available at openneuro in the _*.tsv_ files.

## BOLD fMRI preprocessing

Spatially and temporally normalize all BOLD runs before you start. Use `flirt` to register all the individual runs to a single run (can pick any run), this will put all runs in the same space, which is necessary for machine learning to work correctly. Use excerpts from the preprocessing pipeline. Each voxel should then be normalized to mean zero and standard deviation 1. Also consider bandpass filtering the image.

## Samples extraction

After registering all runs, set up machine learning problem by extracting ‘samples’ from the BOLD using the _*.tsv_ stimulus files accompanying each run. Each sample in the training set is a full-brain BOLD signal at a certain time point (the time point 4 seconds after onset of the image).

If use convolutional neural networks, can flatten the BOLD samples into a 2d representation, or keep the original 3d representation, otherwise each sample will just be a 1d vector of size n_voxels (where n_voxels is the number of voxels in the brain). There are many techniques to flatten the brain, consider using the theta, phi components of a spherical transform over the x,y,z coordinates (called the ‘pancake transform’), or can use [freesurfer’s method] (https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferOccipitalFlattenedPatch) for inflating and flattening the cortex.

## Machine learning modelling

After extracting samples (flattened, or otherwise) using the stimulus file, create a model to predict images based on BOLD samples. The model should predict the raw images, but can also predict features of the image or some low-dimensional representation of the images.
* show raw BOLD image of two co-registered runs, and the signal from the same occipital lobe voxel in both images (to show that the images have been spatially normalized correctly).
* create a small figure showing the model, if it is a neural network draw a diagram
* natural image reconstruction
* artificial image reconstruction

Show bar charts representing the average pearson correlation between the reconstructed images and the ground truth images, based on the test set.

This [article](https://www.frontiersin.org/articles/10.3389/fncom.2019.00021/full) may be of interest.
