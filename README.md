## MuSe-Trust Challenge - ACM Multimedia 2020 - prl source code

### Requirements
- Python 3.7
- Tensorflow 2.2.0
- Tensorflow probability
- Tensorflow addons
- Pandas
- NumPy
- Tqdm
- Tabulate

![Overview of our method](overview.png?raw=true "An overview of our method")

### How to run
1. Modifiy path to dataset of MuSe-challenge in *src/configs/configuration.py*.
2. Use *data_generator.py* in *src* folder to generate tfrecords file.
3. Run *train_test.sh* to run the training.
4. To generate test prediction, modified *line 11* and *line 28* in *train_test.sh*. These line specify the folder which contain corresponding checkpoints. We also provide checkpoints for our submission 2 and 3 in *src/checkpoints* folder.

### Results

Evaluation metric: concordance correlation coefficient (CCC) - <img src="https://latex.codecogs.com/gif.latex?%5Csmall%20%5Crho_%7Bc%7D">
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Csmall%20%5Crho_%7Bc%7D%28y%2C%20%5Chat%7By%7D%29%20%3D%20%5Cfrac%7B2%5Csigma%28y%2C%20%5Chat%7By%7D%29%7D%7B%5Csigma%28y%2Cy%29%20&plus;%20%5Csigma%28%5Chat%7By%7D%2C%5Chat%7By%7D%29%20&plus;%20%28%5Cmu-%5Chat%7B%5Cmu%7D%29%5E%7B2%7D%7D%20%3D%20%5Cfrac%7B2%5Crho%28y%2C%5Chat%7By%7D%29%5Csqrt%7B%5Csigma%28y%2C%20y%29%5Csigma%28%5Chat%7By%7D%2C%5Chat%7By%7D%29%7D%7D%7B%5Csigma%28y%2Cy%29%20&plus;%20%5Csigma%28%5Chat%7By%7D%2C%5Chat%7By%7D%29%20&plus;%20%28%5Cmu-%5Chat%7B%5Cmu%7D%29%5E%7B2%7D%7D%20%3D%20%5Cleft%5B1%20&plus;%20%5Cfrac%7B%5Cmathrm%7BMSE%7D%28y%2C%5Chat%7By%7D%29%7D%7B2%5Csigma%28y%2C%5Chat%7By%7D%29%7D%5Cright%5D%5E%7B-1%7D">
  </p>
  
The above equation also describe the relationship between CCC score and MSE (mean square error) which presented in [2]. In this repo, we implemented metrics and losses function based on above function with tensorflow-probability and tensorflow 2.2.0 as in *src/utils.py*.


| Feature  | CCC-devel | CCC-test |
| ------------- | ------------- | ------------- |
| DS + FT + VG  | 0.3193  | **0.3353** |
| DS + FT + Landmarks  | 0.3426  | 0.3259 |
| FT + VG + RA - baseline method [1] | 0.3198 | **0.4128** |
| DS – baseline method [1] | 0.2019 | 0.1701 |
| FT - baseline method [1] | 0.2278 | 0.2549 |
| Ge + FT + V - baseline method [1] | 0.1245 | 0.1695 |
| Ge + FT - baseline method [1] | 0.2296 | 0.2054 |
### References

[1] Lukas Stappen, Alice Baird, Georgios Rizos, Panagiotis Tzirakis, Xinchen Du, Felix Hafner, Lea Schumann, Adria Mallol-Ragolta, Bj ̈orn W. Schuller, Iulia Lefter, Erik Cambria, Ioannis Kompatsiaris: “The 2020 Multimodal Sentiment Analysis in Real-life Media Workshop and Challenge: Emotional Car Reviews in-the-wild”, Proceedings of *ACM-MM 2020, Seattle, United States*, 2020.

[2] Pandit, Vedhas, and Björn Schuller. "The many-to-many mapping between concordance correlation coefficient and mean square error." *arXiv preprint arXiv:1902.05180* (2019).
