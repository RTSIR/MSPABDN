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

# Train MSPABDN denoising network

To train the MSPABDN denoising network, first download the [BSD400 dataset](https://github.com/smartboy110/denoising-datasets/tree/main/BSD400) and save this dataset inside the main folder of this project. Then generate the training data using:

python Generate_Patches_Gray.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the MSPABDN model file using:

python MSPABDN_Gray.py

This will save the 'MSPABDN_Gray.h5' file in the folder 'Pretrained_models/'.
