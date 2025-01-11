# This repo contains tools for extracting any information of a neural network model. 
# during the forward pass. The tools are designed to be used in conjunction with the PyTorch framework.
# the code snippets are adapted from: https://github.com/RuiLiFeng/Rank-Diminishing-in-Deep-Neural-Networks/blob/main/rank_jacobian.py







def extract_patch(images, method='zero_padding', patch_size=16, image_size=224, row_idx=104, col_idx=104):
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
    
    
    if use_cuda:
        images = images.cuda()
    
    # for debugging:
    print(f'image.size: {images.shape}')
    with torch.no_grad():
        if method == 'interpolate':
            # a function that can be used to resize the patch back to the original size.
            preprocess = partial(functional.interpolate, size=(images.size(2), images.size(3)))
            # resize the image to the patch size by interpolation
            images = functional.interpolate(images, size=(patch_size, patch_size))
        elif method == 'zero_padding':
            # this method extract a patch of the image, starting from row_idx and col_idx:
            # to give a patch of size patch_size x patch_size. shape: (batch_size, channels, patch_size, patch_size)
            images = images[:, :, row_idx:row_idx + patch_size, col_idx:col_idx + patch_size]
            padding_size = [(image_size - patch_size) // 2 for _ in range(4)]
            
            # a function that pads the patch with zeros to restore it to the original image_size.
            preprocess = partial(functional.pad, pad=padding_size, value=0.)
        elif method == 'none' or method is None:
            preprocess = None
    if args.debug:
        images = functional.interpolate(images, size=(patch_size, patch_size))
    print(f'input_image.size: {images.shape}')
    return images, preprocess


def compute_jacobian_rank(images, preprocess, sample_idx, save_jacob=False, verbose=False):

    func = partial(net.forward, preprocess=preprocess)
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
            saved_path = osp.join(saved_name)
            torch.save(jacob.detach().cpu(), saved_path, pickle_protocol=4)

    log_print.write('No{:03d}: '.format(sample_idx) + ', '.join(['{:.2f}'.format(rank[-1]) for rank in ranks]) + '\n')


if __name__==''__main__:
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