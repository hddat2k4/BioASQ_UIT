import hashlib
import os
# Hàm tính MD5 từ file
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Hàm đọc MD5 từ file .md5
def read_md5_from_file(md5_file_path):
    try:
        with open(md5_file_path, "r") as f:
            line = f.readline()
            return line.split('=')[1].strip()  # Lấy phần mã MD5 phía trước
    except:
        return None
pub_dir = "pubmed_baseline_2025"
totalfile = 1274

# Giả sử bạn vừa tải xong file pubmed25n1165.xml.gz
xml_file = "pubmed25n1165.xml.gz"
md5_file = "pubmed25n1165.xml.gz.md5"
xml_path = os.path.join(pub_dir, xml_file)
md5_path = os.path.join(pub_dir, md5_file)


lis = []

# Tính toán và kiểm tra
def check_md5(i):
    xml_file = f"pubmed25n{i:04d}.xml.gz"
    md5_file = f"{xml_file}.md5"
    xml_path = os.path.join(pub_dir, xml_file)
    md5_path = os.path.join(pub_dir, md5_file)
    computed = calculate_md5(xml_path)
    expected = read_md5_from_file(md5_path)
    if expected:
        if computed == expected:
            print(f"✅ MD5 khớp cho {xml_file}")
        else:
            print(f"❌ MD5 KHÔNG KHỚP cho {xml_file}")
            print(f"Expected: {expected}")
            print(f"Computed: {computed}")
            lis.append(xml_file)

    else:
        print(f"⚠️ Không tìm thấy file MD5 cho {xml_file}")

# Kiểm tra từ file 1 đến file 1274

for i in range(1, totalfile + 1):
    check_md5(i)

with open("danh_sach_file.txt", "w", encoding="utf-8") as f:
    for item in lis:
        f.write(f"{item}\n")