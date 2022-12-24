# importing the zipfile module
from zipfile import ZipFile


import os

path = "C:/Users/d0g0825/Downloads/"

os.chdir(path)


for file in os.listdir():
    if file.endswith(".zip"):
        file_path = f"{path}\{file}"
        with ZipFile(file_path, 'r') as zObject:

            zObject.extractall(path=f"{path}")
