# MSPABDN
This repo contains the KERAS implementation of "Blind Gaussian Deep Denoiser Network using Multi-Scale Pixel Attention"


# Run Experiments

To test for blind Gray denoising using MSPABDN write:

python Test_gray.py

The resultant images will be stored in 'Test_Results/Gray/'

To test for blind Color denoising using MSPABDN write:

python Test_color.py

The resultant images will be stored in 'Test_Results/Color/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.

# Train MSPABDN gray denoising network

To train the MSPABDN gray denoising network, first download the [BSD400 dataset](https://github.com/smartboy110/denoising-datasets/tree/main/BSD400) and save this dataset inside the main folder of this project. Then generate the training data using:

python Generate_Patches_Gray.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the MSPABDN model file using:

python MSPABDN_Gray.py

This will save the 'MSPABDN_Gray.h5' file in the folder 'Pretrained_models/'.


# Train MSPABDN color denoising network

To train the MSPABDN color denoising network, first download the [CBSD432 dataset](https://github.com/Magauiya/Extended_SURE/tree/master/Dataset/CBSD432) and save this dataset inside the main folder of this project. Then generate the training data using:

python Generate_Patches_Color.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the MSPABDN model file using:

python MSPABDN_Color.py

This will save the 'MSPABDN_Color.h5' file in the folder 'Pretrained_models/'.

# Citation
@inproceedings{thakur2022blind,
  title={Blind Gaussian Deep Denoiser Network using Multi-Scale Pixel Attention},
  author={Thakur, Ramesh Kumar and Maji, Suman Kumar},
  booktitle={2022 IEEE International Conference on Visual Communications and Image Processing (VCIP)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
