from pathlib import Path
import os

root: Path = Path(__file__).parent.parent


def remove_files():
    files_to_remove = []
    for ext in ["pyd", "coverage"]:
        for file in root.rglob(f"*.{ext}"):
            files_to_remove.append(file.absolute().as_posix())

    if len(files_to_remove) > 0:
        print("Removing the following files: ")

        for file in files_to_remove:
            os.remove(file)
            print(f"\t{file}")


if __name__ == '__main__':
    remove_files()
