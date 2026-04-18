# **FOR ALEX**
The two masks are computed using all the IDs for which the corresponding file in `data/makeathon-challenge/labels/train/glads2` were available. They are pickle file, each of which contains:
- a string representing the key, which is the ID 
- the corresponding mask, a `np.ndarray`

I computed two logistic maps. In `binary_masks.pickle` each entry of the array is between 0 and 1. It was fitted on basics binary flags from a subset of the training set. On the other hand `non_binary_masks.pickle` contains entry that span between 0-4, since they were trained on fitting a logistic curve that actually tried to predict confidence probability rather than just a binary feature. I decided not to normalize it but of course feel free to do such. Actually I think that it might be better for you to do such.

In the notebook `logistic_regression_training.ipynb`, you find the complete pipeline that I used to extract the masks. Every relevant method (for computing masks using also `gladl` or `radd` features, for example) should be at the beginning, the end was mostly validation tuning and saving in the correct .geojson format. I know it's messy but I did not have time to make it pretty, and I'm also tired as fuck.