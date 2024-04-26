# Introduction
This project originates from the [E3PO project](https://github.com/bytedance/E3PO)


# Problems in Original ERP Method
We've identified three major problems in the ERP method. The problems will be explained in the following sections.

## Whole Background Transfer Problem
The first problem 
concerns the transfer of the entire background. The ERP 
method preprocesses a background stream with low resolution and always adds it 
to the download list. While this approach is useful for instances when predictions 
fail or tiles haven't arrived at frame time, we believe it is quite wasteful. A 
significant portion of the background stream may go unseen by the viewer, and 
the pixels in this background stream are unnecessary for tiles already sent with a 
high-resolution, resulting in inefficient resource utilization.

## Prediction Setting Problem
The second problem concerns the prediction setting and algorithm. The ERP method uses an exponential smoothing method to predict motion 
immediately following the historical motion and employs it as the motion for the 
future chunk. However, there is a significant time gap between the motion history 
time and the decision time, as well as between the decision time and the arrival 
time, as illustrated in the graph below. With such a large time gap, the prediction 
is highly likely to be incorrect, and exponential smoothing may not accurately 
predict motion over such a distance. Additionally, given the substantial gap 
between the decision time for the next chunk and the actual time for the next 
chunk, this method may not effectively compensate for prediction errors during 
this period.
![erp prediction graph](/docs/erp_prediction_graph.jpg)

## Network Condition Ignorance Problem
The third problem is the disregard for network conditions in the decision step. In 
the ERP method, network conditions are completely overlooked during the 
decision-making process. However, since the arrival time depends on the size of a 
decided chunk, there is a chance that the decided tiles will not arrive before they 
are needed, leading to a drop in overall quality.


# Proposed Solution
## Probability Prediction Model
The first part of our proposed method tackles the prediction setting and algorithm problem. This aspect is adapted from [this paper](https://doi.org/10.1145/3123266.3123291), which utilizes a probability model to calculate the view probability of each tile for a predicted motion. The main theorem behind this approach is derived from experimentation, revealing that the prediction error of yaw, pitch, and roll using the least square method follows a Gaussian Distribution. Consequently, they can calculate the probability of the correctness of an arbitrary orientation using the following equation:
```math
\begin{cases}
P_{yaw}(\alpha)=\frac{1}{\sigma_\alpha\sqrt{2\pi}}\exp\{-\frac{[\alpha-(\hat\alpha+\mu_\alpha)]^2}{2\sigma^2_\alpha}\}, \\
P_{pitch}(\beta)=\frac{1}{\sigma_\beta\sqrt{2\pi}}\exp\{-\frac{[\beta-(\hat\beta+\mu_\beta)]^2}{2\sigma^2_\beta}\}, \\
P_{roll}(\gamma)=\frac{1}{\sigma_\gamma\sqrt{2\pi}}\exp\{-\frac{[\gamma-(\hat\gamma+\mu_\gamma)]^2}{2\sigma^2_\gamma}\}.
\end{cases}
```
```math
P_E(\alpha,\beta,\gamma)=P_{yaw}(\alpha)P_{pitch}(\beta)P_{roll}(\gamma).
```
Using this probability, they can calculate the viewing probability of each point in the spherical space by considering all possible orientations and averaging the probability of each viewport where this point is visible. Subsequently, they can determine the viewing probability of a tile by averaging the viewing probabilities of all points within the tile. This probability can then be utilized to assign tiles with different resolutions.
In our adapted approach, during the preprocess step, we preprocess each chunk into two resolutions: high-resolution, which is identical to the original, and medium-resolution, which is one-fourth of the original resolution.
In the decision stage, we utilize a certain span of historical motion to predict future motion within a specified time frame. These predicted motions are then used to generate the probability of each tile. To simplify the process, each predicted motion undergoes only six orientations: up, down, left, right, front, and back. We then calculate the sum of probabilities for each point in a tile as a variable related to its viewing probability.
For tiles with this variable exceeding a high threshold, we add the high-resolution tile to the download list. For tiles with this variable falling between a medium threshold and the high threshold, we add the medium-resolution tile to the download list. We then remove tiles already sent as high-resolution from the high-resolution list, and tiles already sent as high or medium-resolution from the medium-resolution list. Combining these two lists yields the final decision list.

## Surrounding Background Tile
The second part of our proposed method aims to address the whole background transfer problem. In the preprocess step, we divide the background stream used in the ERP method into six by six tiles, which are the same size as the high-resolution and medium-resolution tiles.
In the decision step, we add the background tiles surrounding the high-resolution and medium-resolution tiles to the download list. The table below provides an example. The red, orange, and blue blocks represent high-resolution tiles, medium-resolution tiles, and background tiles, respectively.
![background table](/docs/background_table.jpg)

## Network Adaption
The third part of our proposed method aims to address the problem of ignoring 
network conditions. In the decision step, after determining the lists for high-resolution, medium-resolution, and background tiles, we utilize the network 
conditions and the expected arrival time (the latest time for motion prediction) to 
calculate the usable size.
We then compare the total size of the current download list with the usable size. If 
the current size exceeds the usable size, we downgrade high-resolution tiles to 
medium-resolution tiles one at a time. If the size still exceeds the usable size after 
all high-resolution tiles have been converted into medium tiles, we then 
downgrade medium tiles to background tiles one at a time.


# License
[GPL 2.0 License](./COPYING)
