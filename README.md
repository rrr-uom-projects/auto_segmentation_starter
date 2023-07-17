# auto_segmentation_starter
Cleaned up version of some 3D auto-segmentation code

# Preliminary instructions
1. Clone this repo
2. create a new python virtual environment: _python -m venv autosegmentation_env_
3. activate the virtual environment: _source autosegmentation_env/bin/activate_
4. install requirements: _pip install -r requirements.txt_

# Next up
0. Convert Dicom to Nifti
1. Combine structures into a single mask
2. use preprocess.py to preprocess the data
3. train using train.py
4. evaluate the results (using Hausdorff distance, mean distance-to-agreement, DSC...)