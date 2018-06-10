# Google Landmark Retrieval Challenge
We apply several traditional and non-traditional models to our Google Landmark Retrieval dataset and make a comparison between them based on speed, performance rate, and simplicity. As we had an established amount of time for project implementation, our main goal was to make our approach perform as fast as possible due to computing capability constraints.

## Why Google Landmark Retrieval Challenge?
Image retrieval is a fundamental problem in computer vision: given a query image, can you find similar images in a large database? This is especially important for query images containing landmarks, which accounts for a large portion of what people like to photograph.

## Algorithms 
* RootSIFT
* HardNet
* Partition Min-hashing

## Tools
* Python 3
* ImageHash
* PIL

## Evaluation 
* Evaluation
* Normalized hamming distance
* Equality Percentage (EP)

## Our own similarity metric, which we considered the right choice for our algorithm.
Here is how it works. Calculate the hamming distance of every pair with the next logic: IF 1 - diff(h11, h21) > t = 1 else 0, where t is a threshold established for particular hash function. 
After comparison of every pair, we get a binary array of the length M. Distance would be the sum of ones in the binary array.

## Authors
* Andrew Pankiv
* Orest Rehusevych
