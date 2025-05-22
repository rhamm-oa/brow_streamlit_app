import os
import glob
import shutil

# Set your base directory
base_dir = r"/home/user/explore_brows/old_cities"  # Change to your path!
output_folder = os.path.join(base_dir, "all_frontal")
os.makedirs(output_folder, exist_ok=True)

cities = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

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

                # Copy *03 images to all_frontal
                for ext in ["jpg", "png"]:
                    img03 = os.path.join(number_path, f"{number_folder}03.{ext}")
                    if os.path.exists(img03):
                        dest = os.path.join(
                            output_folder,
                            f"{number_folder}.{ext}"
                        )
                        shutil.copy2(img03, dest)

                # Delete *01, *02, *04, *05 images
                for suffix in ["01", "02", "04", "05"]:
                    for ext in ["jpg", "png"]:
                        img = os.path.join(number_path, f"{number_folder}{suffix}.{ext}")
                        if os.path.exists(img):
                            os.remove(img)


