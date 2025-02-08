# This repo contains tools for extracting any information of a neural network model. 
# during the forward pass. The tools are designed to be used in conjunction with the PyTorch framework.
# the code snippets are adapted from: https://github.com/RuiLiFeng/Rank-Diminishing-in-Deep-Neural-Networks/blob/main/rank_jacobian.py

import os
# import os.path as osp 

import numpy as np
import functools 
import torch
import torch.nn.functional as functional
from torch.autograd.functional import jacobian
from omegaconf import DictConfig
# from configs import get_args
from core.data import build_data_loader
from core.models import build_model
from core.utils.logger import Logger

def compute_rank_for_list_of_features(
    list_of_features:list = [],#a list of tensors
    extraction_function = None # the callable function to use for rank extraction
        ):
    
    # length of list_of_features:
    num_layers_to_extract = len(list_of_features)
    
    rank_summary_list = []
    
    for layer_idx in range(num_layers_to_extract):
        extraction_results = extraction_function(list_of_features[lay_idx])
        rank_summary_list.append(extraction_results)
        
    assert len(rank_summary_list )== num_layers_to_extract,"failed to extract ranks for some layers. "
    return rank_summary_list
    
    
    


def extract_patch(images: torch.Tensor,
                  config: DictConfig ,
                  method='zero_padding',
                  patch_size=16, 
                  image_size=224,
                  row_idx=104,# only for zero_padding method
                  col_idx=104 # only for zero_padding method
                  ):
    """
    extracts a patch from each image in the batch. Depending on the method, 
    it either resizes the image, extracts a specific patch and pads it, or leaves it unchanged.
    Parameters:
        images: Batch of input images. 
        method: Strategy for patch extraction ('interpolate', 'zero_padding', or None).
        patch_size: Size of the square patch to extract.
        image_size: Original image size (assuming square images).
        row_idx, col_idx: Starting indices for patch extraction.
            
            
    """
    
    
    # if use_cuda:
    #     images = images.cuda()
    
    # for debugging:
    print(f'image.size: {images.shape}')
    with torch.no_grad():
        if method == 'interpolate':
            # a function that can be used to resize the patch back to the original size.
            preprocess = functools.partial(functional.interpolate, size=(images.size(2), images.size(3)))
            # resize the image to the patch size by interpolation
            images = functional.interpolate(images, size=(patch_size, patch_size))
        elif method == 'zero_padding':
            # this method extract a patch of the image, starting from row_idx and col_idx:
            # to give a patch of size patch_size x patch_size. shape: (batch_size, channels, patch_size, patch_size)
            images = images[:, :, row_idx:row_idx + patch_size, col_idx:col_idx + patch_size]
            padding_size = [(image_size - patch_size) // 2 for _ in range(4)]
            
            # a function that pads the patch with zeros to restore it to the original image_size.
            preprocess = functools.partial(functional.pad, pad=padding_size, value=0.)
        
        # not doing anything to the image:
        elif method == 'none' or method is None:
            preprocess = None
    # if args.debug:
    #     images = functional.interpolate(images, size=(patch_size, patch_size))
    print(f'input_image.size: {images.shape}')
    return images, preprocess


def compute_jacobian_rank(net, images, preprocess, sample_idx, save_jacob=False, verbose=False):
    """
    net: 
        The neural network model. It is expected to have a predict method, with output (final_output, features) 
        where features is a list of intermediate layer outputs. These are to be tracked.
    images:
        Batch of input images.
    preprocess:

    
    
    
    
    
    
    """
    
    
    
    func = functools.partial(net.forward, preprocess=preprocess)
    jacobs = jacobian(func, images, strict=True)

    for index, jacob in enumerate(jacobs):
        jacob = jacob.squeeze().reshape(-1, jacob.shape[-3] * jacob.shape[-2] * jacob.shape[-1])
        jacob_rank = torch.matrix_rank(torch.mm(jacob.T, jacob)).item()
        ranks[index].append(jacob_rank)
        if verbose:
            log_print.write('[No{:03d}]: rank={}, abs_mean={:.5f}, shape={}\n'.format(
                sample_idx, jacob_rank, torch.mean(torch.abs(jacob)), jacob.shape))
        if save_jacob:
            saved_name = 'jaco_logs/jacob_{}_{}_{}_{}.pt'.format(args.model, args.data, index, args.method)
            saved_path = os.path.join(saved_name)
            torch.save(jacob.detach().cpu(), saved_path, pickle_protocol=4)

    log_print.write('No{:03d}: '.format(sample_idx) + ', '.join(['{:.2f}'.format(rank[-1]) for rank in ranks]) + '\n')

def main():
    for i, (images, _) in enumerate(data_loader, start=1):

        if i < selected_sample_indices[0]:
            continue

        images, preprocess = extract_patch(images, method='zero_padding' if not args.debug else None)
        compute_jacobian_rank(images, preprocess, i)

        if i >= selected_sample_indices[1]:
            break

if __name__=='__main__':

    # Init
    # args = get_args()
    use_cuda = True

    # Data
    data_loader = build_data_loader(args, args.data, args.imagenet_dir, shuffle=True,
                                    batch_size=1, num_workers=args.num_workers)
    if args.sample_idx is not None:
        selected_sample_indices = [int(item) for item in args.sample_idx.split(',')]
    else:
        selected_sample_indices = [1, 10]
        args.sample_idx = ','.join([str(i) for i in selected_sample_indices])

    # Model
    net = build_model(args.model, args.method, no_epoch=args.epoch_num, use_cuda=use_cuda,
                      pretrained=not args.wo_pretrained, args=args)
    with torch.no_grad():
        test_inputs = torch.rand(1, 3, 224, 224)
        num_layers = len(net(test_inputs.cuda() if use_cuda else test_inputs))
        layers = ['Layer{}'.format(i) for i in range(num_layers)]
    ranks = [[] for i in range(num_layers)]

    # Logging
    log_path = 'logs_jacob/{}_{}/{}-{}_16x16_jacob_rank.txt'.format(
        str(args.data), args.model, args.model, args.sample_idx)
    if args.wo_pretrained:
        log_path = log_path.replace('.txt', '_wo_pretrained.txt')
    log_print = Logger(log_path)
    log_print.write('logger successful\n')
    log_print.write(str(layers) + '\n')

    # Main
    main()

    log_print.write('n_samples is {}: '.format(len(ranks[0])) + ', '.join(
        ['{:.2f}'.format(np.mean(rank)) for rank in ranks]) + '\n')
    log_print.close()
    # # create a random image tensor
    # images = torch.randn(1, 3, 224, 224)
    # patch_size = 16
    # image_size = 224
    # row_idx = 104
    # col_idx = 104
    # method = 'zero_padding'
    # patch, preprocess = extract_patch(images, method, patch_size, image_size, row_idx, col_idx)
    # print(f'patch.size: {patch.size()}')
    # print(f'preprocess: {preprocess}')
    # print(f'patch: {patch}')