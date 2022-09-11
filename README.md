# ape-escape
Ape Escape is a deep learning model trained using the Python Keras frontend over a TensorFlow backend that assesses how risk-prone and aggressively someone playing the 1v1 fighting game Guilty Gear XX Accent Core +R (aka +R) plays based on a provided .ggr file, which contains a replay of a single match. 

**Input Preprocessing:**
.ggr files encode metadata, player inputs, and gamestate snapshots in hexadecimal.
Before data from a .ggr file is inputted into the model, it must be converted into a list of integers, with each byte of data (two hex digits) being used as a single list element. 
Then, the first 136 elements of the list, which are solely used to contain match metadata, are stripped off. 
Since the metadata is not particularly helpful when it comes to assessing whether a player plays in an aggressive and risk-prone manner, we strip it off to avoid giving the model potential confounding variables.

It's worth noting that I don't understand the process by which the game turns inputs into data, nor how to translate the encoded data into inputs.
With this information, it would be possible to utilize smarter preprocessing methods to ensure the model has an easier time finding the patterns we want it to find.

Another thing of note is that decompressing the .ggr files using zlib gives something more comprehensible and structured than the raw data.
However, it's still not clear to me how to translate this data to inputs, so it's not much of an improvement on the preprocessing end.
In addition, a vast portion of the result consists of nothing but 00's, meaning it would be inefficient to process as is. Stripping the excess 00's risks losing hidden information (such as temporal separation) that the model could make use of, so that is also not an acceptable solution.

In the event that I manage to better understand the inner workings of +R and its replay files, I'll surely see progress on these roadblocks.

**Model Structure**
As of now, the model utilizes a 1D convolutional net on top of a GRU recurrent network to process the data efficiently without sacrificing too much accuracy. Since the sequences are thousands of bytes in length, the convolutional net is used to condense the data down to a size that's managable for a RNN.

**Model Performance**
As of now, the model still performs very imprecisely, with a high variance of error. This could be for a number of reasons:
* Poor understanding of the structure of .ggr files
* Insufficient data
* Insufficiently powerful model architecture
Please stay tuned for future improvements.

**Running Ape Escape on .ggr files**
Unfortunately, downloading TensorFlow is currently necessary to run Ape Escape. Once you do, you can run the Python script run_apeescape.py with file names included as 
command line arguments in order to run the model.
*Disclaimer:* As stated earlier, the model is very inconsistent, so prepare for the possibility of an extremely poor prediction.
