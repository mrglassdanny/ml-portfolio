
import os

pgn_path = "C:/Users/d0g0825/Downloads"
combined_pgn_path = "C:/dev/ml-portfolio/zero-system/schneizel-zero/data/all.pgn"

os.chdir(pgn_path)


def read_pgn_file(path):
    with open(path, 'r') as f:
        return f.read()


combined_pgn_file = open(combined_pgn_path, 'a+')

for file in os.listdir():
    if file.endswith(".pgn"):
        file_path = f"{pgn_path}\{file}"
        combined_pgn_file.write(read_pgn_file(file_path) + "\n")

combined_pgn_file.close()
