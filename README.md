## Pioneer Research Project

#### Author: Tom Zhou



This project mainly consists of a custom model for human segmentation which is lightweight and accurate. By adopting **dilated depth-wise separable convolutions** as a way to reduce weights and computation, our encoder-decoder architecture achieves approximately 90 IOU on our test dataset. The model is about 3M large.

In this repo, we only present several not fully trained model weights (actually they can be seen as full models stored in the hdf5 format), due to time and MONEY constraints. Among the hdf5 files, the file named "kaggle_myIOU_weights" is the latest version, trained for about 30 epochs. I personally believe that the model can be further trained to increase its capacity. This will be a priority next. 

Here's a link to a Kaggle Notebook visualizing some of our model's results. This version (6th commit) showcases the model trained for about 30 epochs. The dataset is **not publicly available**, so the notebook is likely not going to be runnable, unless provided with another similar dataset. Apologizes for the inconvenience.

<https://www.kaggle.com/tommzzhou/pioneer-training/output?scriptVersionId=18707224>

Besides the link above, this repo has several jupyter notebooks that I used when developing this model. These are not final versions.
