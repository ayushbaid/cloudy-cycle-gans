'''
Defining image transforms for data
'''
import torchvision.transforms as transforms


def get_fundamental_transforms(im_size, mean_val, std_val):
  '''
  Performs the most basic transforms on input images
  '''

  return transforms.Compose([
      transforms.RandomCrop(size=im_size),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean_val, std=std_val)
  ])
