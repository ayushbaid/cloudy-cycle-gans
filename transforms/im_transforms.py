'''
Defining image transforms for data
'''
import torchvision.transforms as torchtransforms


def get_fundamental_transforms(im_size, mean_val=0, std_val=1):
  '''
  Performs the most basic transforms on input images
  '''

  return torchtransforms.Compose([
      torchtransforms.RandomCrop(size=im_size),
      torchtransforms.ToTensor(),
      torchtransforms.Normalize(mean=mean_val, std=std_val)
  ])
