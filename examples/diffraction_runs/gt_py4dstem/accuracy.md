# Accuracy of the ground-truth positions

## 1. Lattice fit residual RMS (px), 300 patterns, median over patterns
       rms_ours  rms_py4d_s0.5  rms_py4d_s0
angle                                      
1.0      0.4044         0.0877       0.1053
2.0      0.4060         0.0809       0.1022
4.0      0.4045         0.0786       0.0987
all: rms_ours         0.4048, rms_py4d_s0.5    0.0822, rms_py4d_s0      0.1019
py4d_s0 better than ours in 74.0% of patterns; py4d_s0.5 better in 72.7%

## 2. Distance to the measured disk centre at 1 mrad (px), 2927 isolated disks
          d_ours  d_py4d_s0.5  d_py4d_s0
count  2927.0000    2927.0000  2927.0000
mean      0.4996       0.3978     0.3999
std       0.2767       0.3028     0.2989
min       0.0036       0.0005     0.0036
50%       0.4642       0.3335     0.3344
90%       0.8764       0.8311     0.8366
max       1.6644       1.9272     1.8843