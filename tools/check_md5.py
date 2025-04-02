import os
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thư mục lưu file
baseline_folder = "pubmed_baseline_2025"
os.makedirs(baseline_folder, exist_ok=True)

# Base URL
base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

# Cấu hình
start_file = 1
end_file = 1274
max_retries = 5  # số lần thử lại tối đa nếu lỗi
retry_delay = 5  # giây nghỉ giữa các lần thử lại
max_workers = 5  # số luồng tải đồng thời

# Hàm tải 1 file với retry
def download_file(i):
    file_name = f"pubmed25n{i:04d}.xml.gz"
    file_url = base_url + file_name
    file_path = os.path.join(baseline_folder, file_name)

    if os.path.exists(file_path):
        print(f"{file_name} đã tồn tại, bỏ qua...")
        return

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Bắt đầu tải {file_name} (thử lần {attempt})")
            response = requests.get(file_url, stream=True, timeout=(5, 30))
            if response.status_code != 200:
                print(f"Không tìm thấy file: {file_name}")
                return

            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, "wb") as f, tqdm(
                total=total_size,
                desc=f"Tải {file_name}",
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False  # Không giữ thanh tiến trình sau khi xong
            ) as bar:
                last_chunk_time = time.time()
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
                        last_chunk_time = time.time()
                    elif time.time() - last_chunk_time > 10:
                        raise Exception("Timeout khi chờ dữ liệu từ server.")

            print(f"Tải thành công: {file_name}")
            return

        except Exception as e:
            print(f"Lỗi khi tải {file_name} (lần {attempt}): {e}")
            if attempt < max_retries:
                print(f"Chờ {retry_delay} giây trước khi thử lại...")
                time.sleep(retry_delay)
            else:
                print(f"⛔️ Bỏ qua {file_name} sau {max_retries} lần thử.")

# Tải file với ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_file, i) for i in range(start_file, end_file + 1)]
    for future in as_completed(futures):
        _ = future.result()  # đảm bảo bắt được lỗi nếu có
