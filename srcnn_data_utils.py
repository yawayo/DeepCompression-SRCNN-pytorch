import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageFilter


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        # self.dir_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        #self.image_filenames = []
        # for dir in self.dir_filenames:
        #    for image in listdir(dir):
        #        self.image_filenames.append(join(dir, image))

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index]).convert("YCbCr")
        target = input.copy()
        if self.input_transform:
            #input = input.filter(ImageFilter.GaussianBlur(2)).convert("YCbCr")

            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
