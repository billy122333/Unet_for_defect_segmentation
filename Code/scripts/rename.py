# rename the files in the folder
# to the format of 'scratch_img_0001.png'...

import os

path = '/home/frieren/r12921062/Unet/data_zip/scratch_masks'
files = os.listdir(path)
i = 1

for file in files:
    num = file.split('_')[-1].split('.')[0]
    os.rename(os.path.join(path, file), os.path.join(path, 'scratch_mask_' + num + '.png'))

