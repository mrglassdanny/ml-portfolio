
import os
from zipfile import ZipFile

zip_path = "C:/Users/d0g0825/Downloads"

os.chdir(zip_path)

for file in os.listdir():
    if file.endswith(".zip"):
        file_path = f"{zip_path}\{file}"
        with ZipFile(file_path, 'r') as zObject:

            zObject.extractall(path=f"{zip_path}")
