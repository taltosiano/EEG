
import json
import os

# Create the save directory if it doesn't exist
results_dir = './dataAnalysis'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
# Open the JSON file
with open('./data/prep_data.json', 'r') as json_file:
    # Load the JSON data into a Python dictionary
    data = json.load(json_file)

## see histograms of eeg spamlpes len and labels
labels_list = []
samples_len_list = []
hit_list = []
pattern_list = []
gap_list = []
for dict_idx, value in data.items():
    labels_list.append(value['label'])
    samples_len_list.append(len(value['eeg_dat']))
    if value['label'] == 'plain_hit':
        hit_list.append(len(value['eeg_dat']))
    if value['label'] == 'pattern':
        pattern_list.append(len(value['eeg_dat']))
    if value['label'] == 'gap_element':
        gap_list.append(len(value['eeg_dat']))

import collections
import matplotlib.pyplot as plt
# Count the frequency of each string
frequency = collections.Counter(labels_list)
# Get the strings and their corresponding counts
labels, counts = zip(*frequency.items())
# Create a bar plot for the histogram
plt.figure()
plt.bar(labels, counts)
# Add title and labels
plt.title('Frequency of Labels')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.savefig(os.path.join(results_dir, "label_hist.png"))
# Show the plot
#plt.show()
#### samples len:####
plt.figure()
custom_bin_edges = range(min(samples_len_list), max(samples_len_list) + 2)
plt.hist(samples_len_list, bins=custom_bin_edges)
plt.title('Frequency of Samples len.')
plt.xlabel('Samples len.')
plt.ylabel('Frequency')
plt.savefig(os.path.join(results_dir, "samples_len_hist.png"))
plt.subplot(1, 3, 1)
custom_bin_edges = range(min(hit_list), max(hit_list) + 2)
plt.hist(hit_list, bins=custom_bin_edges)
plt.title('Frequency of PLAIN HIT')
plt.xlabel('Samples len.')
plt.ylabel('Frequency')
plt.tight_layout()
plt.subplot(1, 3, 2)
custom_bin_edges = range(min(pattern_list), max(pattern_list) + 2)
plt.hist(pattern_list, bins=custom_bin_edges)
plt.title('PATTERN')
plt.xlabel('Samples len.')
plt.ylabel('Frequency')
plt.tight_layout()
plt.subplot(1, 3, 3)
custom_bin_edges = range(min(gap_list), max(gap_list) + 2)
plt.hist(gap_list, bins=custom_bin_edges)
plt.title('GAP_ELEMENT')
plt.xlabel('Samples len.')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "categories_len_hist.png"))


