from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random


class ClassificationHandler(Dataset):
    """Dataloader for multi-labels obtained from segmentation mask

    """

    def __init__(self, files, transform=None, Inference=True, save_folder=None):
        """
        The two functions must be overwritten
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = files
        self.transform = transform
        self.Inference=Inference
        self.save_folder =save_folder

    def __len__(self):
        return len(self.files)


    def random_fov(self,image, seg):

        h,w, c = np.shape(image)

        xcenter = h//2
        ycenter = w//2

        offset1 = random.randint(int(xcenter*0.2), int(xcenter*0.6))
        offset2 = random.randint(int(xcenter*0.7), int(xcenter*0.9))

        crop_slice = image[xcenter - offset1:xcenter + offset1, ycenter - offset2:ycenter + offset2,:]
        crop_seg = seg[xcenter - offset1:xcenter + offset1, ycenter - offset2:ycenter + offset2]


        return crop_slice,crop_seg

    def __getitem__(self, idx):

        filename = self.files[idx]

        data_ = np.load(filename)
        data = data_['data']
        y = data_['label']
        seg = data_['seg']

        data= np.stack([data,data,data], axis=-1)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data[:, :, 0] = (data[:, :, 0] - mean[0]) / std[0]
        data[:, :, 1] = (data[:, :, 1] - mean[1]) / std[1]
        data[:, :, 2] = (data[:, :, 2] - mean[2]) / std[2]

        if self.Inference==False and not('verse19' in filename):
            flag = random.randint(0, 1)
            if flag:
                data,seg = self.random_fov(data,seg)

        data= data.transpose((2, 0, 1))
        tdata = self.transform(data)
        return tdata, y

