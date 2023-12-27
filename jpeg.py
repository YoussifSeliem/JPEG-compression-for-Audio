import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct, idct
import heapq
from collections import defaultdict
import wave
import array

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
                    # print(x,zigzag_list[x])
                    x+=1
                    #result.append(matrix[j][i - j])
            else:  # Odd rows go down
                for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                    result[j][i - j]=zigzag_list[x]
                    # print(x,zigzag_list[x])
                    x+=1
                    #result.append(matrix[j][i - j])


        # for k, value in enumerate(zigzag_list):
        #     result[i][j] = value
        #     if (i + j) % 2 == 0:  # Even sum, move right/up
        #         if j < cols - 1:
        #             j += 1
        #         else:
        #             i += 1
        #     else:  # Odd sum, move left/down
        #         if i < rows - 1:
        #             i += 1
        #         else:
        #             j += 1

        return result

huffman_codes_list=[]
encoded_data_list=[]
decimal_data_list=[]

def sound_process(file_name,data):
    first_channel =data[:,0]
    second_channel =data[:,1]
    first_channel,second_channel=padding_channels(first_channel,second_channel)
    return sample_rate,convert_channels_2d(first_channel,second_channel)

def padding_channels(first_channel,second_channel):
    while not np.sqrt(first_channel.size).is_integer():
        first_channel=np.append(first_channel,0)
        second_channel=np.append(second_channel,0)
    return first_channel,second_channel

def convert_channels_2d(first_channel,second_channel):
    matrix_size = int(np.sqrt(len(second_channel)))
    first_channel_2d = np.reshape(first_channel, (matrix_size, matrix_size))
    second_channel_2d = np.reshape(second_channel, (matrix_size, matrix_size))
    return first_channel_2d,second_channel_2d

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
    with open(output_file, 'a') as file:
        for value in text:
            file.write(f"{value} ")

input_file = "input_audioTest.wav"
output_file = "compressed_output"
sample_rate, data = wavfile.read(input_file)
print(data.shape)
sample_rate,(first_channel,second_channel) = sound_process(input_file,data)

print(first_channel)
print(second_channel.shape)


block_size = (8, 8)

# Reshape the matrix into blocks
# Calculate the number of blocks along each dimension
num_blocks_row = first_channel.shape[0] // block_size[0]
num_blocks_col = first_channel.shape[1] // block_size[1]

print(num_blocks_row)
print(num_blocks_col)

# Reshape the matrix into blocks
first_channel_blocks = first_channel[:num_blocks_row * block_size[0], :num_blocks_col * block_size[1]].reshape(num_blocks_row, block_size[0], num_blocks_col, block_size[1])
second_channel_blocks = second_channel[:num_blocks_row * block_size[0], :num_blocks_col * block_size[1]].reshape(num_blocks_row, block_size[0], num_blocks_col, block_size[1])
#blocks = first_channel[:num_blocks_row * block_size[0], :num_blocks_col * block_size[1]].reshape(num_blocks_row, block_size[0], num_blocks_col, block_size[1])
# Now, 'blocks' is a 4D NumPy array where each block has dimensions (8, 8)
# Accessing a specific block, for example, the block at position (0, 0):

print(second_channel_blocks.shape)




def encode(channel):
    #encoded_data_string=''
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):

            curr_block =channel[i, :, j, :]
            dct_block = dct(dct(curr_block.T, norm='ortho').T, norm='ortho').astype(np.int16)
            quantized_block = (dct_block//quantization_matrix).astype(np.int16)
            zigzag_block=zigzag_order(quantized_block)
            encoded_data, huffman_codes = huffman_coding(zigzag_block)
            huffman_codes_list.append(huffman_codes)
            encoded_data_list.append(encoded_data)
            
            #encoded_data_string = encoded_data_string + encoded_data
            #print(type(encoded_data))
            

            # first_block = second_channel_blocks[0, :, 0, :]
            # first_block = second_channel_blocks[0, :, 0, :]
            # print(first_block.shape) ########## wrong
            # print(first_block)

            # print(dct_block)

            # print("QUAN")
            # print(quantized_block)


            # print(zigzag_block)






            # Example usage:
            # matrix = [
            #     [1, 2, 3, 4, 5, 6, 7, 8],
            #     [9, 10, 11, 12, 13, 14, 15, 16],
            #     [17, 18, 19, 20, 21, 22, 23, 24],
            #     [25, 26, 27, 28, 29, 30, 31, 32],
            #     [33, 34, 35, 36, 37, 38, 39, 40],
            #     [41, 42, 43, 44, 45, 46, 47, 48],
            #     [49, 50, 51, 52, 53, 54, 55, 56],
            #     [57, 58, 59, 60, 61, 62, 63, 64]
            # ]

            # zigzag_result = zigzag_order(matrix)

            binary_chunks = [encoded_data[i:i+8] for i in range(0, len(encoded_data), 8)]

            # Convert each binary chunk to a decimal number
            decimal_numbers = [int(chunk, 2) for chunk in binary_chunks]

            # print("Binary String:", encoded_data)
            # print("Decimal Numbers:", decimal_numbers)
            decimal_data_list.append(decimal_numbers)
            # print("Huffman Codes:", huffman_codes)
            # print("Encoded Data:", encoded_data)

            # encoded_data, huffman_codes = huffman_coding(zigzag_result)
            # print(i)
            # print(j)
            # print("Huffman Codes:", huffman_codes_list)
            # print("Encoded Data:", encoded_data_list)
    #write_to_file(output_file,huffman_codes_list)
    #print(huffman_codes_list)
    #write_to_file(output_file,decimal_data_list)

    #print(encoded_data_list)
    print(len(encoded_data_list))
    #print(encoded_data_string)
    write_to_file(output_file,decimal_data_list)
encode(first_channel_blocks)
encode(second_channel_blocks)
            
def decode():
    # Example usage:
    file_path = output_file

    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the entire file content into a string
        file_content = file.read()
        
        # Alternatively, read the file line by line into a list
        # file_content_lines = file.readlines()

    # Print or process the content as needed
    print("File Content:")
    # file_content = ast.literal_eval("[" + file_content + "]")
    file_content=file_content.replace(' ','')
    file_content=file_content.replace('[','')
    returened_blocks=file_content.split(']')
    # print(returened_blocks[0].split(','))
    reconstructed_matrix = np.zeros((num_blocks_row * block_size[0], num_blocks_col * block_size[1]))
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            decoded_data = huffman_decoding(encoded_data_list[num_blocks_col*i+j], huffman_codes_list[num_blocks_col*i+j])
            returned_blocks = reverse_zigzag_order(decoded_data,8,8)
            unquantized_block = returned_blocks * quantization_matrix
            idct_block=idct(idct(unquantized_block.T, norm='ortho').T, norm='ortho').astype(np.int16)
            #print(idct_block)
            block = idct_block
            row_start, row_end = i * block_size[0], (i + 1) * block_size[0]
            col_start, col_end = j * block_size[1], (j + 1) * block_size[1]
            reconstructed_matrix[row_start:row_end, col_start:col_end] = block
    reconstructed_channel = []
    print(reconstructed_matrix)
    print(reconstructed_matrix.shape)
    reconstructed_matrix = reconstructed_matrix.astype(np.int16)
    for i in range(num_blocks_row*8):
        for j in range(num_blocks_col*8):
            reconstructed_channel.append(reconstructed_matrix[i][j])
    #reconstructed_channel = reconstructed_matrix.tolist()
    # print(reconstructed_channel)
    print(len(reconstructed_channel))
    print("testing")
    audio_bytes = array.array('h', reconstructed_channel).tobytes()
    with wave.open("reconstructed_sound.wav", 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.setnframes(len(reconstructed_channel))
        wav_file.writeframes(audio_bytes)
    print(reconstructed_matrix)
    print(reconstructed_matrix.shape)
    #print(decoded_data.shape)
        # print("Decoded Data:", returned_blocks)


    
    # # Example usage:
    # original_matrix = reverse_zigzag_order(decoded_data, 8, 8)
    # print("Original Matrix:")
    # print(original_matrix)

    # unquantized_block=original_matrix*quantization_matrix
    # print(unquantized_block)

    # idct_block=idct(idct(unquantized_block.T, norm='ortho').T, norm='ortho').astype(np.int16)
    # print(idct_block)

decode()
# decimal_numbers=['2131242389573489056734809657340985347','243143215423563464634']
# binary_chunks_reverse = [bin(number)[2:] for number in decimal_numbers]
# print(binary_chunks_reverse)