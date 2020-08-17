# Visual cortex decoding (mind reading)

Acquiring BOLD fMRI images from the brain of a participant who is viewing images or watching a movie sets up an interesting machine learning problem.

Given the brain activity (BOLD fMRI signal in visual cortex) and a set of images displayed to the subject, train a model that can reconstruct what a person is seeing for some unknown images, based on the BOLD signal in their visual cortex.

[Article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) describing dataset, read materials and methods to understand dataset.

Dataset: [Deep Image Reconstruction](https://openneuro.org/datasets/ds001506/versions/1.3.1) (from Openneuro)

You may use only a single subject to reduce the data size (But feel free to extend it to all subjects). In the natural image presentation experiments, the stimulus images are named as 'n03626115_19498', where
'n03626115' is ImageNet/WorNet ID for a synset (category) and '19498' is image ID. You can download the geometric shape images and alphabetical shape images at openneuro.

All information needed to extract events is available at openneuro in the *.tsv files.

## BOLD fMRI preprocessing

You will want to spatially and temporally normalize all BOLD runs before you start. You should use flirt to register all the individual runs to a single run (can pick any run), this will put all runs in the same space, which is necessary for machine learning to work correctly. You’ll probably want to use excerpts from the preprocessing pipeline. Each voxel should then be normalized to mean zero and standard deviation 1. You may also consider bandpass filtering the image.

## Samples extraction

After registering all runs, set up your machine learning problem by extracting ‘samples’ from the BOLD using the *.tsv stimulus files accompanying each run. Each sample in the training set is a full-brain BOLD signal at a certain time point (the time point 4 seconds after onset of the image).
