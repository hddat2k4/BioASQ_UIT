import os
import gzip
import shutil

def compress_pkl_to_gz(input_path, output_path):
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def batch_compress(folder_path, start=922, end=941):
    for i in range(start, end + 1):
        file_name = f"slide3_{i:04d}.pkl"
        input_file = os.path.join(folder_path, file_name)
        output_file = input_file + ".gz"

        if os.path.exists(input_file):
            compress_pkl_to_gz(input_file, output_file)
            print(f"✅ Đã nén: {file_name} → {file_name}.gz")
        else:
            print(f"❌ Không tìm thấy file: {file_name}")

# --- Chạy script ---
folder = r"D:\output\infloat"
batch_compress(folder)
