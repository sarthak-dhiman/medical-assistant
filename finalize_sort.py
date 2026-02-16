import os
import shutil
from pathlib import Path

SORTED_DIR = Path(r"D:\Disease Prediction\Dataset\teeth\test\sorted_by_class")
TEST_DIR = Path(r"D:\Disease Prediction\Dataset\teeth\test")

def main():
    if not SORTED_DIR.exists():
        print("Sorted directory not found.")
        return

    # Move Class_0 to Caries
    class_0 = SORTED_DIR / "Class_0"
    if class_0.exists():
        target = TEST_DIR / "Caries" # Renaming to likely class name
        if target.exists():
            print(f"Target {target} already exists, merging...")
            for f in class_0.iterdir():
                try:
                    shutil.move(str(f), str(target / f.name))
                except:
                    pass
            class_0.rmdir()
        else:
            shutil.move(str(class_0), str(target))
        print("Moved Class_0 to Caries")

    # Move other classes
    for item in SORTED_DIR.iterdir():
        if item.is_dir():
            target = TEST_DIR / item.name
            if target.exists():
                print(f"Target {target} already exists, merging...")
                for f in item.iterdir():
                    try:
                        shutil.move(str(f), str(target / f.name))
                    except:
                        pass
                item.rmdir()
            else:
                shutil.move(str(item), str(target))
            print(f"Moved {item.name} to {TEST_DIR}")

    # clean up
    try:
        SORTED_DIR.rmdir()
        print("Removed sorted_by_class directory")
    except:
        pass

if __name__ == "__main__":
    main()
