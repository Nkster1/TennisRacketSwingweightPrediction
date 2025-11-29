# GAN Training Results

*Comparison of different loss configurations on tennis racket dataset (n=210).*

*Lower values indicate better performance. Bold indicates best result per metric.*

|                          |   Baseline | Diversity Loss   | Correlation Loss   |   Diversity + Correlation |
|:-------------------------|-----------:|:-----------------|:-------------------|--------------------------:|
| Wasserstein Distance     |     5.2745 | **0.8861**       | 1.3775             |                    2.0975 |
| Correlation Difference   |     2.3368 | 0.5379           | **0.3286**         |                    0.4087 |
| Mean Absolute Difference |     3.2546 | **0.4717**       | 0.7164             |                    1.1698 |
| Std Absolute Difference  |     5.1155 | **0.1320**       | 0.5128             |                    0.7586 |
