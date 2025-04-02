import os
import requests
import time
from tqdm import tqdm

# Thư mục lưu file
baseline_folder = "pubmed_baseline_2025"
os.makedirs(baseline_folder, exist_ok=True)

# Base URL
base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

# Cấu hình
start_file = 1197
end_file = 1274
max_retries = 5  # số lần thử lại tối đa nếu lỗi
retry_delay = 5  # giây nghỉ giữa các lần thử lại

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
            # Tăng timeout lên: 10 giây cho kết nối, 60 giây cho nhận dữ liệu
            response = requests.get(file_url, stream=True, timeout=(10, 60))
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
                leave=True
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            print(f"✅ Tải thành công: {file_name}")
            return  # tải xong thì thoát khỏi vòng lặp retry

        except Exception as e:
            print(f"⚠️ Lỗi khi tải {file_name} (lần {attempt}): {e}")
            if attempt < max_retries:
                print(f"🔁 Chờ {retry_delay} giây trước khi thử lại...")
                time.sleep(retry_delay)
            else:
                print(f"⛔️ Bỏ qua {file_name} sau {max_retries} lần thử.")

# Chạy lần lượt từ file start đến end
for i in range(start_file, end_file + 1):
    download_file(i)
import os
import requests
import time
from tqdm import tqdm

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
            # Tăng timeout lên: 10 giây cho kết nối, 60 giây cho nhận dữ liệu
            response = requests.get(file_url, stream=True, timeout=(10, 60))
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
                leave=True
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            print(f"✅ Tải thành công: {file_name}")
            return  # tải xong thì thoát khỏi vòng lặp retry

        except Exception as e:
            print(f"⚠️ Lỗi khi tải {file_name} (lần {attempt}): {e}")
            if attempt < max_retries:
                print(f"🔁 Chờ {retry_delay} giây trước khi thử lại...")
                time.sleep(retry_delay)
            else:
                print(f"⛔️ Bỏ qua {file_name} sau {max_retries} lần thử.")

# Chạy lần lượt từ file start đến end
for i in range(start_file, end_file + 1):
    download_file(i)
