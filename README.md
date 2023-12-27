# JPEG-compression-for-Audio

> this algorithm takes the idea of JPEG compression on images to be applied to audio

## procedure

#### Encoding
- The wav file used is mono (with one channal)
- the samples of the audio file is represented in list
- convert it to square matrix 
- you need padding with zeros to make sure that the matrix is square
- the rest will be the process close to JPEG for image
- start by divide the matrix into blocks of shape (8,8)
- then apply DCT to get the frequenct coefficients on each block
- divide each block by the quantization matrix (for this reason this algorithm is lossy)
- then you can apply any lossless compression method (we applied huffman coding)


#### Decoding
- you will do the previous steps but reversed
- start bt reversing the lossless compression
- then multiply the resulting blocks by the quantization matrix
- then apply IDCT(Inverse DCT) on the blocks
- now you have a blocks, we will combine them to form a big square matrix
- turn this channel into a list
- remove the padded zeros to get the pure data
- write this data into wave file
- Congratz, you reconstructed the wav file successfully