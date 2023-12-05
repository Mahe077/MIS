import re

# Read the contents of the text file
with open('/hdd/2019CS077/Code/v0/checkpoints/maps_cyclegan_rgb_v1/loss_log.txt', 'r') as file:
    content = file.read()

# Regular expression pattern to extract time values
pattern = r"time: (\d+\.\d+)"

# Find all matches of the pattern in the content
matches = re.findall(pattern, content)

# Convert the extracted time values to float and calculate the total time
total_time = sum(float(time) for time in matches)

print(f"Total time: {total_time:.3f}")
