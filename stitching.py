'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    
    #TODO: Add your code here. Do not modify the return and input arguments.
    


    # # mask out the people (artifacts in the foreground)

    # # mask = torch.ones_like(img)
    # # mask[:, :, region_with_person] = 0
    # for i, im in enumerate(imgs):
    #     print(f'image shape = {imgs[im].shape}')
    #     if i == 0:
    #         Lmask = imgs[im] 
    #         Lmask[:, 60:210, 160:215] = 0 # [C, H, W]
    #     show_image(Lmask)
    #     if i == 1:
    #         Rmask = imgs[im]
    #         Rmask[:, 150:, 400:600] = 0
    #     # show_image(Rmask)
    #     # show_image(imgs[im])


    


    # make a new picture that is the cropped right half of image A. 
    for idx, im in enumerate(imgs):
        if idx == 0:
            A_rh = imgs[im][:, :, 220:]
    imgs["A_rh"] = A_rh # append to the dicitonary of images. 

    # # show images
    # for im in imgs:
    #     print(im)
    #     show_image(imgs[im])
    

    # make a list of the image tensors 
    # and normalize to float32 values between 0 and 1
    # and of shape [batch_size, color, height, width] for processing by the Neural Network:
    
    
    
    my_imgs = []
    
    for img_str in imgs:
        if img_str == "A_rh":
            continue # skip, to be appended as first picture
        

        img_tensor = imgs[img_str]
        # 0 is min, 255 is max, 
        # dividing by 255 normalizes values in that range to between 0 and 1
        img_tensor = img_tensor.to(torch.float32) / 255.0
        my_imgs.append(img_tensor.unsqueeze(0))

    
    # my_imgs.reverse() # try reversing order to prefer the left image
    # make the first image A_rh:
    img_tensor = imgs["A_rh"].to(torch.float32) / 255.0
    my_imgs = [img_tensor.unsqueeze(0)] + my_imgs # put the cropped picture first

    # stitched images must have the same height.
    # I will resize all images uniformly so that they have the same height as the tallest height image
    my_imgs = resize_images(my_tensor_list= my_imgs) # resize all uniformly to have the same height

    # t1_1.png = A (person dressed in black)
    # t1_2.png = B (person dressed in blue)
    # A_rh = right half of A (no people)

    # ArB_list = [my_imgs[0], my_imgs[2]]
    ArB_list = [my_imgs[2], my_imgs[0]] # This worked! Niether guy is in the picture!
    # A_list = [my_imgs[1]] not needed

    KF = K.feature
    IS = K.contrib.ImageStitcher(KF.LoFTR(pretrained="outdoor"), estimator="ransac")
    with torch.no_grad():
        # feed the list of image tensors to the stitcher
        # and return img as the ouput tensor
        # img = IS(*my_imgs)  # [B=1, C, H, W] float32
        img = IS(*ArB_list)
    img = img.squeeze() # [C, H, W] float32
    img = img * 255 # scaled back to whole numbers
    img = img.to(torch.uint8) # whole numbers are now uint8

    show_image(img) 


    
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap







def resize_images(my_tensor_list):
    # Assume tensors in list are of shape [B, C, H, W]
    # And they are normalized float32 values between 0 and 1
    my_H_list = [tensor.size(2) for tensor in my_tensor_list]
    target_H = max(my_H_list)

    resized = []

    F = torch.nn.functional
    for tensor in my_tensor_list:
        # print(f'tensor.shape = {tensor.shape}')
        _, _, H, W  = tensor.shape # _ for Batch and Color 
        my_scale = int(target_H / H) # make sure both new Height and Width are ints
        new_W = int(W * my_scale)

        tensor = F.interpolate(tensor, 
                               size=(target_H, new_W), 
                               mode='bilinear', 
                               align_corners=False)
        resized.append(tensor)

    return resized






'''
A stitch B -> AB where the man in black is still there
B stitch A -> BA where the man in blue is still there

Image Stitcher uses the first image supplied as "the ground truth". 
Let's segment the right half of A, where the man in black and blue are not there 
and use that as the ground truth

Then Adding B to Ar would yeild most of the scene without either persons. 
Adding A to ArB would yeild other missing details. 
'''


# def pad_tensors_with_zeros(my_tensor_list, top_list= None):
#     # Assume tensors in list are of shape [B, C, H, W]
#     # Assume my_tensor_list is the same length as top_list
#     if top_list is None:
#         top_list = [True] * len(my_tensor_list)

#     # find the height of the tallest tensor
#     largest_H = 0
#     # largest_idx = 0
#     for idx, my_tensor in enumerate(my_tensor_list):
#         if my_tensor.size(2) > largest_H:
#             largest_H = my_tensor.size(2)
#             # largest_idx = idx

#     # compensate other tensors according to the list
#     # some pictures might have more information 
#     # relating to the scene on the bottom or top of the image,
#     # so that is why I chose to make it a list of bools to deal with more than 2 pictures.

#     padded_tensor_list = []
#     for idx, my_tensor in enumerate(my_tensor_list):
#         padded_tensor_list.append(apply_black_mask(largest_H= largest_H,
#                                                    my_tensor= my_tensor,
#                                                    top= top_list[idx]))
        
#     return padded_tensor_list
        

# def apply_black_mask(largest_H, my_tensor, top):
#     # Assume tensors in list are of shape [B, C, H, W]

#     H_delta = largest_H - my_tensor.size(2)
#     # print(f'H_delta = {H_delta}')
#     W = my_tensor.size(3)
#     if H_delta == 0:
#         return my_tensor # no need to change
#     else:
#         my_mask = torch.zeros((1, 3, H_delta, W))
#         if top ==  True: # put mask on top
#             masked_tensor = torch.cat((my_mask, my_tensor), dim= 2)
#         else: # put mask on bottom
#             masked_tensor = torch.cat((my_tensor, my_mask), dim= 2)

#     return masked_tensor
    

