import os
import numpy as np
import cv2
import albumentations
from PIL import Image
import torch
from torch.utils.data import Dataset


class Sun360CompBase(Dataset):
    def __init__(self,
                 data_csv, data_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=182, 
                 coord=False,
                 no_crop=False,
                 no_rescale=False
                 ):
        self.n_labels = n_labels
        self.data_csv = data_csv
        self.data_root = data_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths]
        }
        self.coord = coord
        if self.coord:
            print("Cylinderical coordinate for 360 image.")

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size, # Sun360 images are 256x512
                                                                 interpolation=self.interpolation)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image", "masked_image": "image", "binary_mask": "image"})
                
            self.preprocessor = self.cropper
        self.no_crop = no_crop
        self.no_rescale = no_rescale

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        
        if not self.no_rescale and self.size is not None: # Default True. False when refine net training.
            image = self.image_rescaler(image=image)["image"]
        
        if False:
            # random scale
            random_h = torch.randint(256,512,(1,))
            image = cv2.resize(image, (random_h*2, random_h), interpolation = cv2.INTER_AREA)

        # Masking here
        masked_image, binary_mask = self.masking(image) # Make input image with holes.
        if not self.coord:
            if self.size is not None:
                processed = self.preprocessor(image=image)
            else:
                processed = {"image": image}
        else: # when using coord (coord=True)
            # Change for cylinderical coord
            h,w,_ = image.shape
            #coord = np.arange(h*w).reshape(h,w,1)/(h*w) * 2 - 1 # -1~1 # old style. arange is insufficient
            coord = np.tile(np.arange(h).reshape(h,1,1), (1,w,1)) / (h-1) * 2 - 1 # -1~ 1
            # sin, cos
            coord = self.add_cylinderical(coord) # -1 ~ 1
            # rotation augmentation 
            image, coord, masked_image, binary_mask = self.rotation_augmentation(image, coord, masked_image, binary_mask)

            if not self.no_crop and self.size is not None: # self.no_crop = True when training Transformer with 1:2 images, or icip(256x512)
                processed = self.cropper(image=image, coord=coord, masked_image=masked_image, binary_mask=binary_mask)
            else:
                processed = {"image": image, "coord": coord, "masked_image": masked_image, "binary_mask": binary_mask}

            example["coord"] = processed["coord"]
            example["masked_image"] = (processed["masked_image"]/127.5 - 1.0).astype(np.float32)
            example["binary_mask"] = processed["binary_mask"]
            

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32) 
        example["concat_input"] = np.concatenate((example["masked_image"], example["coord"], example["binary_mask"]), axis=2)
        return example

    def rotation_augmentation(self, im, coord, masked_im, binary_mask):
        split_point = torch.randint(0, im.shape[1],(1,)) # w 
        #split_point = im.shape[1] // 2
        im = np.concatenate( (im[:,split_point:,:], im[:,:split_point,:]), axis=1 )
        coord = np.concatenate( (coord[:,split_point:,:], coord[:,:split_point,:]), axis=1 )
        masked_im = np.concatenate( (masked_im[:,split_point:,:], masked_im[:,:split_point,:]), axis=1 )
        binary_mask = np.concatenate( (binary_mask[:,split_point:,:], binary_mask[:,:split_point,:]), axis=1 )
        return im, coord, masked_im, binary_mask

    def masking(self, im):
        h, w, c = im.shape
        binary_mask = np.zeros((h,w,1))
        # random mask position
        margin_h = int( (180 - torch.randint(70, 95, (1,)) ) / 360 * h )
        #print(margin_h, (h - margin_h))
        binary_mask[margin_h:(h - margin_h), int(w/4):int(w/4)*3, :] = 1.
        canvas = np.ones_like(im) * 127.5
        canvas = im * binary_mask + canvas * (1 - binary_mask)
        return canvas, binary_mask

    def add_cylinderical(self, coord):
        h, w, _ = coord.shape
        sin_img = np.sin(np.radians(np.arange(w) / w * 360))
        sin_img[np.abs(sin_img) < 1e-6] = 0
        sin_img = np.tile(sin_img, (h,1))[:,:,np.newaxis]
        cos_img = np.cos(np.radians(np.arange(w) / w * 360))
        cos_img[np.abs(cos_img) < 1e-6] = 0
        cos_img = np.tile(cos_img, (h,1))[:,:,np.newaxis]
        return np.concatenate((coord, sin_img, cos_img), axis=2)
        

# Change here for SUN360 dataset
class Sun360CompTrain(Sun360CompBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False):
        super().__init__(# set data_csv
                         #data_csv="data/sun360_train.txt",
                         data_csv="data/sun360_1024b_train.txt", #defalut
                         #data_csv="data/laval_1024_train.txt",
                         # set data_root
                         #data_root="/home/ubuntu/local/outdoor512a/train_B",
                         data_root="/home/ubuntu/tmp_local/outdoor1024b/train", #defalut
                         #data_root="/home/ubuntu/tmp_local/laval1024/train",
                         # set args through a configs yaml.
                         size=size, random_crop=random_crop, interpolation=interpolation, 
                         coord=coord, no_crop=no_crop, no_rescale=no_rescale)

class Sun360CompValidation(Sun360CompBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False):
        super().__init__(# set data_csv
                         #data_csv="data/sun360_val.txt",
                         data_csv="data/sun360_1024b_val.txt", #defalut
                         #data_csv="data/laval_1024_val.txt",
                         # set data_root
                         #data_root="/home/ubuntu/local/outdoor512a/test_B",
                         data_root="/home/ubuntu/tmp_local/outdoor1024b/test", #defalut
                         #data_root="/home/ubuntu/tmp_local/laval1024/test",
                         # set args through a configs yaml.
                         size=size, random_crop=random_crop, interpolation=interpolation, 
                         coord=coord, no_crop=no_crop, no_rescale=no_rescale)