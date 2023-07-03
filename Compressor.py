from PIL import Image
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
RESOLUTION = 8  # known in advance


# read and display an image from EX1
def load_gray_bmp(path):
    img = Image.open(path).convert('L')
    # img.show()
    # print(img.size)
    return np.array(img)


# scaling
# Scale each element of a matrix from [0, 2^(K − 1)] to [−1/2, 2^(K−1) − 1/2^K], where K is the resolution in bits.
def scale_image(matrix: npt.ArrayLike):
    k = RESOLUTION
    # max2-min2/max1-min1
    # scale_factor = (((2**(k - 1) - 1 )/ (2 ** k)) - (-1 / 2)) / (2 ** (k - 1))
    # img = (matrix * scale_factor) - 0.5
    # Image.fromarray(img).show()
    # return img
    # return (matrix - (2 ** (k - 1) - 1)) / (2 ** (k - 1))
    return (matrix - (2 ** (k - 1) - 1)) / (2 ** k)  # true on blocks
    # return (matrix * scale_factor) - 0.5


# Scale each element of a matrix from [0, 2^(K − 1)] to [−1/2, 2^(K−1) − 1/2^K], where K is the resolution in bits.
def unscale(matrix: npt.ArrayLike):  # todo fix !!!!!!!!
    k = RESOLUTION
    # +0.5 * max1-min1/max2-min2
    # unscale_factor = (2 ** (k - 1)) / ((2 ** (k - 1) - 1 / (2 ** k)) - (-1 / 2))
    # return (matrix + 0.5) * unscale_factor
    return (matrix * (2 ** k)) + (2 ** (k - 1) - 1)


# image to blocks todo very slow!!

# todo fit to size d if not possible
def bin_matrix(matrix, d):
    m, n = matrix.shape
    block_structure = []

    for row_start in range(0, m, d):
        for col_start in range(0, n, d):
            sub_matrix = matrix[row_start:row_start + d, col_start:col_start + d]
            # print(sub_matrix.shape)
            block_structure.append(sub_matrix)
    out = np.array(block_structure).reshape((int(m / d), int(n / d), d, d))
    return out


# todo replace
def un_bin_matrix(block_structure):
    rows = len(block_structure)
    cols = len(block_structure[0])
    d = block_structure[0][0].shape[0]

    matrix = np.empty((rows * d, cols * d))

    for i in range(rows):
        for j in range(cols):
            # matrix[i * d:(i + 1) * d, j * d:(j + 1) * d] = IDCT(block_structure[i][j]) #inverse DCT
            matrix[i * d:(i + 1) * d, j * d:(j + 1) * d] = block_structure[i][j]  # inverse DCT

    return matrix


# DCT discrete cosine transform
def DCT(matrix: npt.ArrayLike):
    M, N = matrix.shape
    dct_output = np.empty_like(matrix, dtype=np.float64)
    c = np.empty((M, N))
    c[0, 0] = (1 / np.sqrt(M)) * (1 / np.sqrt(N))
    c[0, 1:] = (1 / np.sqrt(M)) * np.sqrt(2 / N)
    c[1:, 0] = (1 / np.sqrt(N)) * np.sqrt(2 / M)
    c[1:, 1:] = np.sqrt(2 / M) * np.sqrt(2 / N)
    for p in range(M):
        for q in range(N):
            # alpha_p = 1 / np.sqrt(M) if p == 0 else np.sqrt(2 / M)
            # alpha_q = 1 / np.sqrt(N) if q == 0 else np.sqrt(2 / N)

            # if not inverse:
            # dct_output[p, q] = alpha_p * alpha_q * np.sum(  # MxN * Mx1 * 1xN
            dct_output[p, q] = c[p, q] * np.sum(  # MxN * Mx1 * 1xN
                matrix * np.cos((np.pi * p / (2 * M)) * (2 * np.arange(M) + 1)[:, np.newaxis]) *
                np.cos((np.pi * q / (2 * N)) * (2 * np.arange(N) + 1)[np.newaxis, :]))

            # else:
            #     dct_output[p, q] = np.sum(
            #         alpha_p * alpha_q *
            #         matrix * np.cos((np.pi * p / (2 * M)) * (2 * np.arange(M) + 1)[:, np.newaxis]) *
            #         np.cos((np.pi * q / (2 * N)) * (2 * np.arange(N) + 1)[np.newaxis, :]))
    # print(dct_output)

    # dct_output *= np.sqrt(2 / (M* N))
    # dct_output = c * dct_output
    # print(dct_output)
    return dct_output


def IDCT(matrix):
    M, N = matrix.shape
    dct_output = np.empty_like(matrix, dtype=np.float64)

    c = np.empty((M, N))
    c[0, 0] = (1 / np.sqrt(M)) * (1 / np.sqrt(N))
    c[0, 1:] = (1 / np.sqrt(M)) * np.sqrt(2 / N)
    c[1:, 0] = (1 / np.sqrt(N)) * np.sqrt(2 / M)
    c[1:, 1:] = np.sqrt(2 / M) * np.sqrt(2 / N)
    # print(c)
    for m in range(M):
        for n in range(N):
            # for p in range(M):
            #     for q in range(N):
            #         alpha_p = 1 / np.sqrt(M) if p == 0 else np.sqrt(2/M)
            #         alpha_q = 1 / np.sqrt(N) if q == 0 else np.sqrt(2/N)
            #         dct_output[m, n] += alpha_p * alpha_q * matrix[p, q] * np.cos(((2*m + 1)*p*np.pi) / (2*M)) * np.cos(((2*n + 1)*q*np.pi) / (2*N))
            # p_col =  np.arange(M)[:,np.newaxis]
            dct_output[m, n] = np.sum(
                matrix * c * np.cos((((2 * m + 1) * np.pi) / (2 * M)) * np.arange(M)[:, np.newaxis]) * np.cos(
                    (((2 * n + 1) * np.pi) / (2 * N)) * np.arange(N)[np.newaxis, :]))

    return dct_output


# quantize the DCT coefficients divide by delta outputs integer quantization indices
def quantize_coefficients(DCT, delta):
    return np.rint(DCT / delta)


# inverse quantization t
def inverse_quantized_coefficients(Q, delta):
    return Q * delta


#
# # DPCM fot dc coeffecients
# # Xi,j DC vvalue of block i,j
# def prediction_error():
#     e_ij = X_ij - X_ij_hat
#     Q_ij = quantize_coefficients(e_ij,delta)
#
#


def zigzag_order(block):
    N = len(block)
    if not all(len(row) == N for row in block):
        raise ValueError("Input block is not a square matrix")

    vector = []
    for s in range(2 * N - 1):
        if s % 2 == 0:
            i = min(s, N - 1)
            j = max(0, s - N + 1)
            while i >= 0 and j < N:
                vector.append(block[i][j])
                i -= 1
                j += 1
        else:
            i = max(0, s - N + 1)
            j = min(s, N - 1)
            while i < N and j >= 0:
                vector.append(block[i][j])
                i += 1
                j -= 1
    return vector
    # N = len(block)
    # if not all(len(row) == N for row in block):
    #     raise ValueError("Input block is not a square matrix")
    #
    # vector = []
    # i, j = 0, 0
    # for _ in range(N * N):
    #     vector.append(block[i][j])
    #     if (i + j) % 2 == 0:  # Moving up-right
    #         if j == N - 1:
    #             i += 1
    #         elif i == 0:
    #             j += 1
    #         else:
    #             i -= 1
    #             j += 1
    #     else:  # Moving down-left
    #         if i == N - 1:
    #             j += 1
    #         elif j == 0:
    #             i += 1
    #         else:
    #             i += 1
    #             j -= 1
    # return vector


def inverse_zigzag_order(vector):
    N = int(len(vector) ** 0.5)
    if N * N != len(vector):
        raise ValueError("Input vector size is not consistent with a square matrix")

    block = [[0 for _ in range(N)] for _ in range(N)]
    block = np.zeros((N, N))
    idx = 0
    for s in range(2 * N - 1):
        if s % 2 == 0:
            i = min(s, N - 1)
            j = max(0, s - N + 1)
            while i >= 0 and j < N:
                block[i][j] = vector[idx]
                idx += 1
                i -= 1
                j += 1
        else:
            i = max(0, s - N + 1)
            j = min(s, N - 1)
            while i < N and j >= 0:
                block[i][j] = vector[idx]
                idx += 1
                i += 1
                j -= 1
    return block


def mean_squared_error(X, Y): #both are of size MxN
    return np.mean((X - Y) ** 2)


# performance measure
def peak_signal_to_noise_ratio(X, Y):
    mse = mean_squared_error(X, Y)
    A = (2 ** RESOLUTION) - 1   # maximal value
    psnr_db = 20 * np.log10(A / np.sqrt(mse))
    return psnr_db


def calculate_compression_rate(original_size, compressed_size):
    return compressed_size / original_size


# encoder that does scaling ,blocking ,DCT

def encoder(matrix, block_size,delta):  # outputs DCT coefficients  typical d=8 or 16
    blocks = bin_matrix(scale_image(matrix), block_size)
    op_DCT = np.vectorize(DCT, signature='(m,n)->(m,n)')
    dct_blocks = op_DCT(blocks)
    quantized=quantize_coefficients(dct_blocks,delta)
    return quantized


# decoder that takes a dct co's and outputs the original image
def decoder(encoded_img,delta):
    inv_quant_blocks= inverse_quantized_coefficients(encoded_img,delta)
    op_IDCT = np.vectorize(IDCT, signature='(m,n)->(m,n)')
    idct_blocks=op_IDCT(inv_quant_blocks)
    decoded_img = unscale(un_bin_matrix(idct_blocks))
    return decoded_img


if __name__ == '__main__':
    # read image
    matrix = load_gray_bmp("Mona-Lisa.bmp")
    origin_mat = matrix.astype('float64')
    ###############################################################
    # compress with block size 8 and 16 with different delta values
    N= [8,16]
    # delta = [0.5,0.1,0.01,0.001]
    delta = np.linspace(0.001, 0.5, num=10)
    # PSNR_8=[]
    # PSNR_16=[]
    PSNRs = [[],[]]
    deltas=[]
    for i in range(2):
        for d in delta:
            reconstructed_img = decoder(encoder(origin_mat, N[i],d),d)
            psnr = peak_signal_to_noise_ratio(origin_mat,reconstructed_img)
            # if 20 <= psnr <= 35:
            PSNRs[i].append(psnr)
            deltas.append(d)
        plt.plot(deltas,PSNRs[i],label=f'N={N[i]}')
        deltas=[]

    # plt.scatter(delta, PSNRs[0], marker='o', color='blue')
    # plt.scatter(delta, PSNRs[1], marker='x', color='red')
    # theoretical_data = [10*np.log10(12/(d**2)) for d in delta]
    # filtered_data = [(d, psnr) for d, psnr in zip(delta, theoretical_data) if 20 <= psnr <= 35]
    # filtered_deltas, filtered_psnr = zip(*filtered_data)
    # plt.plot(filtered_deltas, filtered_psnr, label='Theoretical')
    plt.xlabel('Quantization Step Size (delta)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.title('PSNR vs. Quantization Step Size for N=8 and N=16')
    plt.show()
    ###############################################################3