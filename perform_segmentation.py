# Author: German A. Garcia Ferrando 
# Contact me: www.github.com/gergf

import argparse
import numpy as np 
import nibabel as nib
from scipy import ndimage
from os.path import dirname
from subprocess import call
import matplotlib.pyplot as plt 

import tensorflow as tf

#############################################################
IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 1 

# Name of the tensors in the meta-graph.
GRAPH_TENSORS = {
    'x_in':  'inputs/x_in:0',
    'training_phase': 'hyperparameters/training_phase:0',
    'probs': 'inference/probability_maps/probs:0' 
}

SAVE_PATH  = './net/convnet'
#############################################################

"""
    Labels each connected region of the image. 
    Returns:
        The label of the biggest connected region
"""
def get_biggest_connected_region(labeled_img, num_objects):
    regions_size = np.zeros(num_objects+1)
    for i in range(1,num_objects+1): 
        regions_size[i] = (np.count_nonzero(labeled_img == i))
    return np.argmax(regions_size)

"""
    Set all the minor regions to background and the biggest to 1.
    Returns: 
        A numpy array with the new segmentation.
"""
def remove_minor_regions(labeled_img, biggest_reg_lab):
    f = np.vectorize(lambda x: 1 if x == biggest_reg_lab else 0)
    return f(labeled_img)

"""
    This does not work in Windows Systems. 
"""
def get_file_name(path):
    return path.split('/')[-1]

"""
    Saves the visualization as png in the specified fied path. 
"""
def save_visualization(segmentation, original_image, path_to_output, alpha=0.5):
    f = plt.figure()
    a = f.add_subplot(131)
    a.imshow(original_image, cmap='gray')
    a.set_title('image')
    a = f.add_subplot(132)
    a.imshow(segmentation, cmap='gray')
    a.set_title('model segmentation')
    a = f.add_subplot(133)
    a.imshow(original_image, cmap='gray')
    a.imshow(segmentation, alpha=alpha)
    a.set_title('visualization')
    plt.savefig(path_to_output)

"""
    Saves the segmentations as a NifTi image in the specified path. 
"""
def save_nifti_segmentation(segmentation, original_nifti_image, path_to_output):
    nifti_seg = nib.Nifti1Image(segmentation, original_nifti_image.affine, original_nifti_image.header)
    nib.save(nifti_seg, path_to_output)

"""
    Performs xRay femur image segmentation by using a pre-trained convolutional neural network. 
"""
def femur_extraction(path_to_input, path_to_output, path_to_pp_sample, path_to_visualizations, verbose):
    # Preprocess data # 
    call(['python3', './preprocess/prepare_sample.py', path_to_input, path_to_pp_sample])

    # Load preprocessed data  #
    nifti_input_image = nib.load(path_to_pp_sample)
    model_input = nifti_input_image.get_data()
    model_input = np.reshape(model_input, (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) # NHWC format 

    # Build the model #
    with tf.Session() as sess:
        # Restore the model #
        saver = tf.train.import_meta_graph(SAVE_PATH + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./net/'))
        # Recover graph tensors # 
        graph = tf.get_default_graph()
        x_in = graph.get_tensor_by_name(GRAPH_TENSORS['x_in']) # Placeholder for the input 
        training_phase = graph.get_tensor_by_name(GRAPH_TENSORS['training_phase']) # Placeholder for Batch Normalization
        probs = graph.get_tensor_by_name(GRAPH_TENSORS['probs']) # Tensor which stores the result of performing the inference
        # Feed model and perform inference # 
        prob_map = sess.run(probs, feed_dict={x_in: model_input, training_phase: False})        
        # Get labels from the probability maps #
        raw_segmentation = np.argmax(prob_map[0], axis=2)

    # Get connected regions # 
    labeled, nr_objects = ndimage.label(raw_segmentation)
    biggest_reg_lab = get_biggest_connected_region(labeled, nr_objects)
    segmentation = remove_minor_regions(labeled, biggest_reg_lab)
    
    # Save result #
    if verbose:
        save_visualization(segmentation, nifti_input_image.get_data(), path_to_visualizations) 
    save_nifti_segmentation(segmentation, nifti_input_image, path_to_output)

#############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs xRay femur image segmentation by using a pre-trained convolutional neural network.' +
        'The model generates two files, one NifTi file, which is the segmentation of the sample; and a PNG file, which is a preview of the segmentation.')
    parser.add_argument('-i',   '--input' , help='Path to load the input sample (.nii.gz).', type=str, required=True)
    parser.add_argument('-o',  '--output', help='Path to save the output sample. Take note that this path must end with ".nii.gz"', default='./output.nii.gz', type=str)
    parser.add_argument('-v', '--verbose', help='If True, the script generates a visualization of the results. (Default Fasle)', default=False, type=bool)
    args = parser.parse_args()

    path_to_input = args.input
    path_to_output = args.output
    
    # Check paths # 
    if len(path_to_input) < 7 or path_to_input[-7:] != '.nii.gz' :
        exit('The input file must be a NifTi image (.nii.gz)')
    
    if len(path_to_output) < 7 or path_to_output[-7:] != '.nii.gz':
        exit('The output file must be a NifTi image (.nii.gz)')

    # Generate results paths # 
    in_file_name = get_file_name(path_to_input)
    dir_path_output = dirname(path_to_output) + '/'
    path_to_visualizations = dir_path_output + 'visualizations'
    path_to_pp_sample = dir_path_output + 'preprocess_' + in_file_name

    femur_extraction(path_to_input, path_to_output, path_to_pp_sample, path_to_visualizations, args.verbose)
