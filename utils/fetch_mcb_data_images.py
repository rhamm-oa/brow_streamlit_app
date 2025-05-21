import os
import glob

# Set this to the path where 'old_city' is located
base_dir = r"/home/user/explore_brows/old_cities"

cities = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
print(f"Found {len(cities)} cities, which are {cities} in {base_dir}")
result = []

for city in cities:
    images_dir = os.path.join(base_dir, city, "Images")
    if not os.path.exists(images_dir):
        continue
    for date_folder in os.listdir(images_dir):
        date_path = os.path.join(images_dir, date_folder)
        if not os.path.isdir(date_path):
            continue
        for tone in ["dark", "light", "medium"]:
            tone_path = os.path.join(date_path, tone)
            if not os.path.isdir(tone_path):
                continue
            for number_folder in os.listdir(tone_path):
                number_path = os.path.join(tone_path, number_folder)
                if not os.path.isdir(number_path):
                    continue
                # Look for XXXX03.jpg inside each number folder
                pattern = os.path.join(number_path, f"{number_folder}03.jpg")
                matches = glob.glob(pattern)
                for match in matches:
                    result.append(match)

# Print all found images (full paths)
for path in result:
    print(path)

# Optionally, copy them to a new folder:
import shutil

output_folder = os.path.join(base_dir, "all_03_images")
os.makedirs(output_folder, exist_ok=True)

for src in result:
    filename = os.path.basename(src)
    dest = os.path.join(output_folder, f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(src)))).replace(' ', '_')}_{filename}")
    shutil.copy2(src, dest)