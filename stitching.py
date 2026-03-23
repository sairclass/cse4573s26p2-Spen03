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

    # print(f'imgs = {imgs}')

    # for im in imgs:
    #     print("in for loop")
    #     print(f'im = {im}')
    #     # show_image(imgs[im])

    '''
    Find overlap array by finding features and comparing them 
    against features of the other images. 
    Assuming 20% overlap == True (1) and <20% overlap == False (0)
    Then the overlap array will be an N x N (N = num of pics) one-hot array
    where element (i, j) will be True if image i overlaps at least 20% with image j.   
    '''
    # resize:
    resized_img_dict = resize_images_dict(my_img_dict= imgs)
    # find overlap matrix and usable images for the panorama
    overlap, usable_imgs_dict = get_overlap_matrix(img_dict= resized_img_dict)
    
    '''
    normalizing not necessary for LoFTR
    '''
    # make a list of usable image tensors normalized and with batch dimension
    # my_imgs = []
    # for img_str in usable_imgs_dict:
    #     img_tensor = usable_imgs_dict[img_str]
    #     my_imgs.append(img_tensor)

    # make a list of usable image tensors normalized and with batch dimension
    my_imgs = []
    for img_str in usable_imgs_dict:
        img_tensor = usable_imgs_dict[img_str]
        # 0 is min, 255 is max, 
        # dividing by 255 normalizes values in that range to between 0 and 1
        img_tensor = img_tensor.to(torch.float32) / 255.0
        # my_imgs.append(img_tensor.unsqueeze(0))
        # usable_imgs already in ([B, C, W, H]) from resize_images_dict
        my_imgs.append(img_tensor) 



    # stitch panorama from usable images
    KF = K.feature
    IS = K.contrib.ImageStitcher(KF.LoFTR(pretrained="outdoor"), estimator="ransac")
    with torch.no_grad():
        # feed the list of image tensors to the stitcher
        # and return img as the ouput tensor
        # img = IS(*my_imgs)  # [B=1, C, H, W] float32
        img = IS(*my_imgs)
    img = img.squeeze() # [C, H, W] float32
    img = img * 255 # scaled back to whole numbers
    img = img.to(torch.uint8) # whole numbers are now uint8

    show_image(img)


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



def resize_images_dict(my_img_dict):
    my_img_tensors = []
    for img_str in my_img_dict:
        # needs to be in ([B, C, H, W]) for resize_images()
        my_img_tensors.append(my_img_dict[img_str].unsqueeze(0)) 

    resized_tensors = resize_images(my_tensor_list= my_img_tensors)

    resized_dict = {}

    for i, img_str in enumerate(my_img_dict):
        resized_dict[img_str] = resized_tensors[i]

    return resized_dict

'''
A stitch B -> AB where the man in black is still there
B stitch A -> BA where the man in blue is still there

Image Stitcher uses the first image supplied as "the ground truth". 
Let's segment the right half of A, where the man in black and blue are not there 
and use that as the ground truth

Then Adding B to Ar would yeild most of the scene without either persons. 
Adding A to ArB would yeild other missing details. 
'''


def get_overlap_matrix(img_dict, found_feat_thresh=0.2):
    # images in img dict are all of size ([B, C, H, W])
    overlap = torch.zeros((len(img_dict), len(img_dict)))
    
    KF = K.feature
    matcher = KF.LoFTR(pretrained= "outdoor")

    # img1 = K.image_to_tensor(np.array(Image.open(fname1).convert("RGB"))).float()[None, ...] / 255.0

    for i, img_i_str in enumerate(img_dict): 
        # convert to black and white
        # and normalize values to float32 between 0 and 1
        bw_i = K.color.bgr_to_grayscale(img_dict[img_i_str].float() / 255.0) 
        # print(f'bw_i = {bw_i}')
        # print(f'bw_i shape = {bw_i.shape}')
        for j, img_j_str in enumerate(img_dict):
            bw_j = K.color.bgr_to_grayscale(img_dict[img_j_str].float() / 255.0)
            # print(f'bw_j = {bw_j}')
            # print(f'bw_j shape = {bw_j.shape}')

            
            compare_dict = {
                "image0" : bw_i, # always pass image0 and image1, no other names
                "image1" : bw_j
            }

            # compare_dict = {
            #     "image0" : img_dict[img_i_str], # always pass image0 and image1, no other names
            #     "image1" : img_dict[img_j_str] # LoFTR only works on grayscale
            # }
            with torch.no_grad():
                matches = matcher(compare_dict)

            match_confidence = matches["confidence"] # tensor of feature match confidence scores
            # print(f'match_confidence.shape = {match_confidence.shape}')
            # filter out low confidence scores
            # find percentage of high confidence scores still left out of total features
            
            # extract number of features from the tensor
            total_features = match_confidence.shape[0] 
            # number of features that have a high confidence match
            conf_thresh = 0.9
            high_conf_matches = (match_confidence > conf_thresh).sum().item()

            high_conf_percentage = high_conf_matches / total_features

            # print(f'high_conf_percentage ({i}, {j}) = {high_conf_percentage}')

            if high_conf_percentage >= found_feat_thresh:
            # if torch.linalg.norm(match_confidence) >= conf_thresh:
                overlap[i, j] = 1



    '''
    # image 1 does not match with image two or three
    # but image 2 and 3 match
    ([[1, 0, 0],
      [0, 1, 1],
      [0, 1, 1]])
      
    ->

    ([[0, 0, 0],
      [0, 0, 1],
      [0, 1, 0]])

    for row in tensor:
        if sum of row > 0, 
        this tensor matched with at least one other image and is usable for panoramas

    '''
    overlapping_other = overlap - torch.eye(len(img_dict)) # subtract matches with own image
    usable_imgs_dict = {}
    for i, img_i_str in enumerate(img_dict):
        if torch.sum(overlapping_other[i]) > 0:
            usable_imgs_dict[img_i_str] = img_dict[img_i_str]


    print(f'overlap = {overlap}')

    return overlap, usable_imgs_dict