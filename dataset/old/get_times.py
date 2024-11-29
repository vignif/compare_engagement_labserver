from pathlib import Path
from bag_analyzer_get_times import run
import time
# Replace 'your_folder_path' with the path to your folder

folder_path = 'UE-HRI/bags/'

# Create a Path object for the folder
folder = Path(folder_path)

# Use the glob pattern to find all .bag files in the folder
bag_files = folder.glob('*.bag')

# Print the list of .bag files
for file in bag_files:
    print(file.stem)
    start_time = time.perf_counter()
    run(filename=file.stem)
    # Record the end time
    end_time = time.perf_counter()

    # Calculate the duration in seconds
    duration = end_time - start_time

    # Print the duration
    print(f"Execution time: {duration:.6f} seconds")