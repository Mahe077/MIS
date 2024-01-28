import re
import csv

# Read the contents of the text file
with open('/hdd/2019CS077/TestCode/v0/checkpoints/maps_cyclegan_rgb_v3_5000_arnold_scrambler/loss_log.txt', 'r') as file:
    content = file.read()

# Regular expression pattern to extract relevant information
pattern = r"\(epoch: (\d+), iters: (\d+), time: ([\d.]+)\) D_A: ([\d.]+) G_A: ([\d.]+) Cyc_A: ([\d.]+) D_B: ([\d.]+) G_B: ([\d.]+) Cyc_B: ([\d.]+) new_loss: ([\d.]+)"

# Find all matches of the pattern in the content
matches = re.findall(pattern, content)

# Define CSV output file
output_file = '/hdd/2019CS077/TestCode/v0/checkpoints/maps_cyclegan_rgb_v3_5000_arnold_scrambler/loss_log.csv'

# Write the matches to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header
    csvwriter.writerow(['epoch', 'iters', 'time', 'D_A',
                       'G_A', 'Cyc_A', 'D_B', 'G_B', 'Cyc_B', 'new_loss'])

    # Write data
    csvwriter.writerows(matches)

print(f"CSV file '{output_file}' created successfully.")
