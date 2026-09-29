# Image forgery detection based on noise inspection: analysis and refinement of the Noisesniffer method

Images undergo a complex processing chain from the moment light reaches the camera’s sensor until the final digital image is delivered. Each of its operations leaves traces on the noise model which enable forgery detection through noise analysis. In this article we describe the Noisesniffer method. This method estimates for each image a background stochastic model which makes it possible to detect local noise anomalies characterized by their number of false alarms. We improve on the original formulation of the method by introducing a region growing algorithm to detect local deviations from the background model. 

Usage:
python Noisesniffer.py input.png -w w -b b -n n -m m -W W
  
input.png	Input image to analyse
-w w		Block size (default: 3)
-b b		Number of blocks per bin (default: 20000)
-n n		Percentile of blocks with the lowest energy in low frequencies (default: 0.1)
-m m		Percentile of blocks with the lowest standard deviation (default: 0.5)
-W W		Cell size for NFA computation (region growing) (default: 100)

A typical execution is as follows:
python Noisesniffer.py test.png -w 3 -b 20000 -n 0.1 -m 0.5 -W 100

Outputs:
output_distributions.png: input image with painted distributions.
output_mask.png: forgery detection results.
NFA.txt: NFA of the detected regions.


LICENSE

This project is licensed under the Apache License 2.0.


