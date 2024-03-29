import os

# 指定目录路径
directory_path = "/home/wangjie/PycharmProjects/TTS/recipes/st-cmds/st-cmds"

# 遍历目录中的所有文件
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, "r") as file:
            file_contents = file.read()
            if "g" in file_contents:
                print(f"File '{filename}' contains 一 character: {file_contents}")
