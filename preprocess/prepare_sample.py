# Author: German A. Garcia Ferrando 
# Contact me: www.github.com/gergf

# Python dependencies 
import os 
import sys 
import nibabel as nib 

# Math 
import numpy as np 
from scipy.stats import iqr

def check_relative_path(path): 
    """
        It adds './' to the begining of the path if it is linux or '.\' when the OS is windows. 
        If the paths starts with '.', '\' or '/', it returns the path as provided. 

        Args: 
            path: Path to check (String)

        Returns: 
            The completed path 
    """
    if path[0] not in ('.', '/', '\\'):
        if sys.platform == "win32":
            return '.\\' + path 
        else:
            return './' + path 
    else:
        return path 

def _resize_sample(source_file_path, target_file_path, _new_dim = '256x256'):
    """
    	Resizes a sample to _new_dim. 
    	This methods uses ResampleImage from the ANTs library, so you need to have it in your path environment. 
    	Note: 
    		The methods only works with two-dimensional images. 
    """
    from subprocess import call
    call(["ResampleImage", '2', source_file_path, target_file_path, _new_dim, "1", "0"])

def _robust_zscore(source_file_path, target_file_path):
    """
    	Performs a zScore to the image using the median and the IQR of that image. 
    """
    img      = nib.load(source_file_path)
    img_data = img.get_data()
    new_img  = (img_data - np.median(img_data, axis=None)) / iqr(img_data)
    nib.save(nib.Nifti1Image(new_img, img.get_affine(), img.get_header()), target_file_path)

def prepare_sample(source_file_path, target_file_path): 
    # Resize the image #
    _resize_sample(source_file_path, target_file_path)
    # Perform a zScore # 
    _robust_zscore(target_file_path, target_file_path)

if __name__ == '__main__': 
    # Check if the input # 
    if len(sys.argv) != 3:
        exit('Invalid number of arguments. Usage: \npython3 prepare_sample.py /path/to/read/sample.nii.gz /path/to/store/preprocessed/sample.nii.gz')

    # Read paths #
    source_file_path = check_relative_path(sys.argv[1])
    target_file_path = check_relative_path(sys.argv[2])

    # Check if path is absolute or relative
    valid_start_symb = ['.', '/', '\\'] # Relative, and absolute for Linux and Win 
    if (source_file_path[0] not in valid_start_symb) or (target_file_path[0] not in valid_start_symb): 
        exit('Invalid path. It must start with "." or ("/", "\\") for Linux/Win.')

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