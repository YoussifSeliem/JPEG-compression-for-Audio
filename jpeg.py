import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct, idct
import heapq
from collections import defaultdict
import wave
import array

# name : youssif tamer seliem
# section : 5
# academic number : 1900494

huffman_codes_list=[]
encoded_data_list=[]
decimal_data_list=[]
# Define a quantization matrix
quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

def zigzag_order(matrix):
    rows, cols = len(matrix), len(matrix[0])
    result = []

    for i in range(rows + cols - 1):
        if i % 2 == 0:  # Even rows go up
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                result.append(matrix[j][i - j])
        else:  # Odd rows go down
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                result.append(matrix[j][i - j])

    return result

def reverse_zigzag_order(zigzag_list, rows, cols):
        result = np.zeros((rows, cols), dtype=int)
        i, j, x = 0, 0, 0
        for i in range(rows + cols - 1):
            if i % 2 == 0:  # Even rows go up
                for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                    result[j][i - j]=zigzag_list[x]
                    x+=1
                    
            else:  # Odd rows go down
                for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                    result[j][i - j]=zigzag_list[x]
                    x+=1

        return result

def sound_process(file_name,data):
    first_channel =data[:]
    first_channel=padding_channels(first_channel)
    return sample_rate,convert_channels_2d(first_channel)

def padding_channels(first_channel):
    while not np.sqrt(first_channel.size).is_integer():
        first_channel=np.append(first_channel,0)
    return first_channel

def convert_channels_2d(first_channel):
    matrix_size = int(np.sqrt(len(first_channel)))
    first_channel_2d = np.reshape(first_channel, (matrix_size, matrix_size))
    return first_channel_2d

def build_huffman_tree(frequencies):
    heap = [[weight, [value, ""]] for value, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0][1:]

def huffman_coding(values):
    frequencies = defaultdict(int)
    for value in values:
        frequencies[value] += 1

    huffman_tree = build_huffman_tree(frequencies)

    huffman_codes = {value: code for value, code in huffman_tree}

    encoded_data = ''.join(huffman_codes[value] for value in values)

    return encoded_data, huffman_codes

def huffman_decoding(encoded_data, huffman_codes):
                reverse_codes = {code: value for value, code in huffman_codes.items()}

                decoded_data = []
                current_code = ''
                for bit in encoded_data:
                    current_code += bit
                    if current_code in reverse_codes:
                        decoded_data.append(np.int16(reverse_codes[current_code]))
                        current_code = ''

                return decoded_data

def write_to_file(output_file, text):
    with open(output_file, 'w') as file:
        for value in text:
            file.write(f"{value} ")

input_file = "input.wav"
output_file = "compressed_output"
sample_rate, data = wavfile.read(input_file)

sample_rate,(first_channel) = sound_process(input_file,data)

added_zeros=len(first_channel)*len(first_channel)-len(data)
block_size = (8, 8)

# Reshape the matrix into blocks
# Calculate the number of blocks along each dimension
num_blocks_row = first_channel.shape[0] // block_size[0]
num_blocks_col = first_channel.shape[1] // block_size[1]

# Reshape the matrix into blocks
channel_blocks = first_channel[:num_blocks_row * block_size[0], :num_blocks_col * block_size[1]].reshape(num_blocks_row, block_size[0], num_blocks_col, block_size[1])

def encode(channel):
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):

            curr_block =channel[i, :, j, :]
            dct_block = dct(dct(curr_block.T, norm='ortho').T, norm='ortho').astype(np.int16)
            quantized_block = (dct_block//quantization_matrix).astype(np.int16)
            zigzag_block=zigzag_order(quantized_block)
            encoded_data, huffman_codes = huffman_coding(zigzag_block)
            huffman_codes_list.append(huffman_codes)
            encoded_data_list.append(encoded_data)

            binary_chunks = [encoded_data[i:i+128] for i in range(0, len(encoded_data), 128)]

            # Convert each binary chunk to a decimal number
            decimal_numbers = [int(chunk, 2) for chunk in binary_chunks]

            decimal_data_list.append(decimal_numbers)

    write_to_file(output_file,decimal_data_list)

def decode():

    reconstructed_matrix = np.zeros((num_blocks_row * block_size[0], num_blocks_col * block_size[1]))

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            decoded_data = huffman_decoding(encoded_data_list[num_blocks_col*i+j], huffman_codes_list[num_blocks_col*i+j])
            returned_blocks = reverse_zigzag_order(decoded_data,8,8)
            unquantized_block = returned_blocks * quantization_matrix
            idct_block=idct(idct(unquantized_block.T, norm='ortho').T, norm='ortho').astype(np.int16)
            block = idct_block
            row_start, row_end = i * block_size[0], (i + 1) * block_size[0]
            col_start, col_end = j * block_size[1], (j + 1) * block_size[1]
            reconstructed_matrix[row_start:row_end, col_start:col_end] = block

    reconstructed_channel = []
    reconstructed_matrix = reconstructed_matrix.astype(np.int16)

    for i in range(num_blocks_row*8):
        for j in range(num_blocks_col*8):
            reconstructed_channel.append(reconstructed_matrix[i][j])
            
    for i in range(added_zeros):
        reconstructed_channel.pop()

    audio_bytes = array.array('h', reconstructed_channel).tobytes()
    with wave.open("reconstructed_sound.wav", 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.setnframes(len(reconstructed_channel))
        wav_file.writeframes(audio_bytes)


encode(channel_blocks)    
decode()
