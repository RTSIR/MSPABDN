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
