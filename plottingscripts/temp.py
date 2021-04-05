import numpy as np
import matplotlib.pyplot as plt

n_bins = 6000
band_start = np.random.randint(100,n_bins,size=1000)

bands = []

for band in band_start:
    octaves = np.random.randint(2,4)
    bands.append([band,min(band*octaves,12000)])

print(bands)

plt.hist(band_start, bins=n_bins)
plt.show()
