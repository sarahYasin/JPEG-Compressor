# This is a sample Python script.
import numpy as np
import scipy
import matplotlib.pyplot as plt
import Compressor
import EX1
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Compressor import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')




###################################################################
# check encoder gradually on mona lisa

    origin_mat = load_gray_bmp("Mona-Lisa.bmp")
    origin_mat=origin_mat.astype('float64')
    Image.fromarray(origin_mat).show()
#########################################################
    all_scaled = Compressor.scale_image(origin_mat)
    # print(all_scaled)
    print("Done Scaling")
##############################################################
    # blocks_scaled= Compressor.bin_matrix(all_scaled, 48)
    blocks_scaled= Compressor.bin_matrix(all_scaled, 8)
    # print(blocks_scaled)
    print("Done binning")
##################################################################
    op_DCT = np.vectorize(Compressor.DCT,signature='(m,n)->(m,n)')
    op_IDCT = np.vectorize(Compressor.IDCT,signature='(m,n)->(m,n)')
    dct_blocks= op_DCT(blocks_scaled)
    print(dct_blocks)
    print("Done DCT")
####################################################################
    # delta = 2
    # op_Quant= np.vectorize(Compressor.quantize_coefficients,signature='(m,n)->(m,n)', excluded=['delta'])
    # quant_blocks = op_Quant(dct_blocks)
    quant_blocks = Compressor.quantize_coefficients(dct_blocks,1)
    print(quant_blocks)
    Image.fromarray(unscale(un_bin_matrix(quant_blocks))).show()
    unquant_blocks = Compressor.inverse_quantized_coefficients(quant_blocks, 1)
    # inverse_quant_blocks= Compressor.inverse_quantize_coefficients(quant_blocks,0.1)
    # x_values = np.ndarray.flatten(dct_blocks)
    # y_values = np.ndarray.flatten(inverse_quant_blocks)
    # print(x_values)
    # plt.plot(x_values, y_values)
    # plt.xlabel('Original Value (x)')
    # plt.ylabel('Inverse-Q(Q(x))')
    # plt.title('Inverse Quantization Plot')
    # plt.grid(True)
    # plt.show()
    # Image.fromarray(un_bin_matrix(quant_blocks)).show()
#####################################################################
    idct_blocks= op_IDCT(unquant_blocks)
    # print(idct_blocks)
    # print(np.array_equal(idct_blocks, blocks_scaled))
    print("Done IDCT")

    are_almost_equal = np.allclose(idct_blocks, blocks_scaled, atol=1e-8)
    print(are_almost_equal)


#############################################################
    unblocked_scaled = Compressor.un_bin_matrix(idct_blocks)
    # print(unblocked_scaled)
    print("Done unbinning")
############################################################
    unblocked_unscaled = Compressor.unscale(unblocked_scaled)
    print("Done unScaling")
    # print(np.array_equal(unblocked_unscaled, origin_mat))
    are_almost_equal = np.allclose(unblocked_unscaled, origin_mat, atol=1e-8)
    print(are_almost_equal)
    Image.fromarray(unblocked_unscaled).show()

################################################################################
# DCT IDCT check on block 8x8 of ones
    # ones_mat = np.ones((8,8))
    # dct_ones = DCT(ones_mat)
    # print(dct_ones)
    # idct_ones = IDCT(dct_ones)
    # print(idct_ones)
    # print(np.array_equal(idct_ones,ones_mat))
    # diff= (idct_ones!=ones_mat)
    # print(diff)
    # are_almost_equal = np.allclose(idct_ones, ones_mat, atol=1e-8)
    # print(are_almost_equal)
    #
######################################################################

    # pre_zigzag = np.arange(1,65).reshape((8, 8)).T
    # print(pre_zigzag)
    # after_zz= zigzag_order(pre_zigzag)
    # print(after_zz)
    # after_uzz = inverse_zigzag_order(after_zz)
    # print(after_uzz)
    # print(pre_zigzag==after_uzz)

##############################################################
    # op_scale = np.vectorize(Compressor.scale_image)
    #
    # op_unscale = np.vectorize(Compressor.unscale)
    #
    # s=op_scale(out2)
    # print(all)
    # u=op_unscale(s)
    # print(s)
    # print(u)
    # print(u==out2)

    # print(out1,out1,out1==out2)

    # mat = np.array([[1, 2, 3, 4],
    #                         [5, 6, 7, 8],
    #                        [9, 10, 11, 12],[13,14,15,16]])
    # a=DCT(mat)
    # b= IDCT(a)
    # print(a)
    # print(b)
    # print(a==b)
    # img = scale_image(load_gray_bmp("Mona-Lisa.bmp"))
    # # verify that idct(dct(A)) = A
    # binned = bin_matrix(img,48)
    # mat = unscale(un_bin_matrix(binned))
    # # print(mat)
    # Image.fromarray(mat).show()
    # print( mat == img)
    # verify that dct(eigenvector)= one non zero coeffeicent

    # def check_non_zero_dct_coeffs(eigenvector):
    #     # Apply the DCT to the eigenvector
    #     dct_result = DCT(eigenvector)
    #     # dct_result2 = scipy.fft.dct(eigenvector, norm='ortho')
    #     # scipy.f
    #
    #     # Find the non-zero coefficients
    #     # non_zero_coeffs = dct_result[np.abs(dct_result) > 1e-8]  # Adjust the threshold as needed
    #     non_zero_coeffs = dct_result[np.abs(dct_result) > 0]  # Adjust the threshold as needed
    #
    #     return non_zero_coeffs


    # Example usage
    # eigenvector = np.array([[1, 2, 3, 4],
    #                [5, 6, 7, 8],
    #                [9, 10, 11, 12]])
    # eigenvector = np.array([[1],[1],[1],[1]])
    # # Compute the non-zero DCT coefficients
    # non_zero_coeffs = check_non_zero_dct_coeffs(eigenvector)
    #
    # print("Eigenvector:", eigenvector)
    # print("Non-zero DCT coefficients:", non_zero_coeffs)

import os
import imageio
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def compress_image(input_image_path, output_image_path, quality):
    with Image.open(input_image_path) as img:
        img.save(output_image_path, format='JPEG', quality=quality)

def get_file_size(file_path):
    return os.path.getsize(file_path)

def calculate_compression_rate(original_size, compressed_size):
    return compressed_size / original_size

# Input image (BMP format)
input_image_path = "input_image.bmp"

# Load the BMP image and get its size in bytes
bmp_image = Image.open(input_image_path)
bmp_size_bytes = os.path.getsize(input_image_path)

# Quality levels to test
quality_levels = range(5, 101, 5)

psnr_values = []
compression_rates = []

for quality in quality_levels:
    # Compress the image at the current quality level and get the compressed size
    compressed_image_path = f"compressed_quality_{quality}.jpg"
    compress_image(input_image_path, compressed_image_path, quality)
    compressed_size_bytes = get_file_size(compressed_image_path)

    # Load the compressed JPG image and convert to numpy array
    compressed_image = imageio.imread(compressed_image_path)
    compressed_image = np.array(compressed_image)

    # Calculate PSNR between original BMP and compressed JPG
    psnr = compare_psnr(bmp_image, compressed_image)
    psnr_values.append(psnr)

    # Calculate compression rate
    compression_rate = calculate_compression_rate(bmp_size_bytes, compressed_size_bytes)
    compression_rates.append(compression_rate)

    # Delete the temporary compressed image file
    os.remove(compressed_image_path)

# Prepare the plot
import matplotlib.pyplot as plt

plt.plot(compression_rates, psnr_values, marker='o')
plt.xlabel('Compression Rate (JPG Size / BMP Size)')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs. Compression Rate')
plt.grid(True)
plt.show()