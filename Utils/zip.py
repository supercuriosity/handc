import os
import zipfile

def zip_folder(folder_path, output_path):
    """
    将指定文件夹压缩为 zip 文件
    :param folder_path: 要压缩的文件夹路径
    :param output_path: 输出 zip 文件路径
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

if __name__ == "__main__":
    # 示例用法
    folder_to_zip = "data/handcap_1758713140.1875231"
    output_zip = "data/zipped/handcap_1758713140.1875231.zip"
    zip_folder(folder_to_zip, output_zip)
    print(f"已将 {folder_to_zip} 压缩为 {output_zip}")