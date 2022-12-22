
import os

path = "C:/Users/d0g0825/Downloads/"

os.chdir(path)


def read_pgn_file(path):
    with open(path, 'r') as f:
        return f.read()


combined_pgn_file = open(
    "c:/dev/ml-portfolio/zero-system/schneizel-zero/data/all.pgn", 'a+')

for file in os.listdir():
    if file.endswith(".pgn"):
        file_path = f"{path}\{file}"
        combined_pgn_file.write(read_pgn_file(file_path))

combined_pgn_file.close()
