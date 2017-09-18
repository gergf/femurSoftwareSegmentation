from scipy.stats import iqr
import nibabel as nib 
import numpy as np 
import os.path 
import sys 


def clean_path(path): 
    if path[0] not in ['.', '/']:
        return './' + path 
    else:
        return path 

"""
	Resizes a sample to _new_dim. 
	This methods uses ResampleImage from the ANTs library, so you need to have it in your path environment. 
	Note: 
		The methods only works with two-dimensional images. 
"""
def _resize_sample(source_file_path, target_file_path, _new_dim = '256x256'):
    from subprocess import call
    call(["ResampleImage", '2', source_file_path, target_file_path, _new_dim, "1", "0"])

"""
	Performs a zScore to the image using the median and the IQR of that image. 
"""
def _robust_zscore(source_file_path, target_file_path):
    img = nib.load(source_file_path)
    img_data = img.get_data()
    new_img = (img_data - np.median(img_data, axis=None)) / iqr(img_data)
    nib.save(nib.Nifti1Image(new_img, img.get_affine(), img.get_header()), target_file_path)

def prepare_sample(source_file_path, target_file_path): 
    # Resize the image #
    _resize_sample(source_file_path, target_file_path)
    # Perform a zScore # 
    _robust_zscore(target_file_path, target_file_path)

if __name__ == '__main__': 
    # Check if the input # 
    if len(sys.argv) != 3:
        exit('Invalid number of arguments. Usage: \npython3 prepare_sample.py path/to/read/sample.nii.gz path/to/store/preprocessed/sample.nii.gz')

    # Read paths #
    source_file_path = clean_path(sys.argv[1])
    target_file_path = clean_path(sys.argv[2])

    # Check if the sample exists #
    if not os.path.isfile(source_file_path) or source_file_path[-6:] != 'nii.gz':
        exit('There was a problem reading the file. Please check the file exists and that its extension is ".nii.gz".')

    # Check target file extension and directory # 
    targetDirPath = os.path.dirname(target_file_path)
    if target_file_path[-6:] != 'nii.gz' or len(target_file_path) < 7 or not os.path.isdir(targetDirPath):
        exit('Invalid name to the target path file. Please check that the name of the file ends with nii.gz extension and \n' + 
             'that it is in a existing directory.')

    # If everything is ok, then run the preprocess # 
    prepare_sample(source_file_path, target_file_path)