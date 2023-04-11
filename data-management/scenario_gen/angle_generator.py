import json
import numpy as np
import bisect


def generate_angle(cumsum, bins, bin_width):
    p = np.random.rand()
    i = bisect.bisect_left(cumsum, p)
    left_edge = bins[i]

    p_offset = np.random.rand()
    offset = bin_width * p_offset

    return left_edge + offset


def generate_1_angle():
    ra = generate_angle(azimuth_cumsum, azimuth_bins, azimuth_bin_width)
    re = generate_angle(elevation_cumsum, elevation_bins, elevation_bin_width)
    return ra, re

# load statistics
with open('real_cam_angle_stats.json', 'r') as f:
    stats = json.load(f)

azimuth_n = np.array(stats['azimuth']['n'])
azimuth_n_norm = azimuth_n / azimuth_n.sum()
azimuth_cumsum = azimuth_n_norm.cumsum()
azimuth_bins = np.array(stats['azimuth']['bins'])
azimuth_bin_width = azimuth_bins[1] - azimuth_bins[0]

elevation_n = np.array(stats['elevation']['n'])
elevation_n_norm = elevation_n / elevation_n.sum()
elevation_cumsum = elevation_n_norm.cumsum()
elevation_bins = np.array(stats['elevation']['bins'])
elevation_bin_width = elevation_bins[1] - elevation_bins[0]

if __name__ == '__main__':
    print('Let\'s try generating some random azimuth and elevation angles')
    rand_azimuths, rand_elevations = [], []
    for i in range(10000):
        rand_azimuth = generate_angle(azimuth_cumsum, azimuth_bins, azimuth_bin_width)
        rand_elevation = generate_angle(elevation_cumsum, elevation_bins, elevation_bin_width)

        print('Azimuth = {}\t\tElevation = {}'.format(rand_azimuth, rand_elevation))
        rand_azimuths.append(rand_azimuth)
        rand_levations.append(rand_elevation)

    '''import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(rand_azimuths, bins=100, label='azimuth')
    axs[1].hist(rand_elevations, bins=100, label='elevation')

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    plt.show()'''