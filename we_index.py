
import os
import pickle
import uuid
import time

import hashlib

def generate_hashed_uuid(vector, raw_uuid):
    h = hashlib.md5()
    h.update(str(vector).encode("utf-8"))
    h.update(str(raw_uuid).encode("utf-8"))
    return str(uuid.UUID(h.hexdigest()))

import weaviate
from tqdm import tqdm
import gc
import subprocess

# --- Thông tin cấu hình ---
input_dir = "embeddings"
container_name = "bioasq-weaviate-1"
collection_name = "PubMedAbstract"
SUB_BATCH_SIZE = 300

# --- Hàm kiểm tra Weaviate đã sẵn sàng sau khi restart ---
def wait_for_weaviate_ready(max_wait_seconds=120):
    start_time = time.time()
    while True:
        try:
            client = weaviate.connect_to_local()
            if client.is_ready():
                return client
        except:
            pass

        if time.time() - start_time > max_wait_seconds:
            raise TimeoutError("Weaviate không khởi động lại đúng thời gian.")
        print("Chờ Weaviate khởi động lại...")
        time.sleep(5)

# --- Bắt đầu tiến trình indexing ---
client = weaviate.connect_to_local()
collection = client.collections.get(collection_name)

#939-941 
for i in tqdm(range(939, 942), desc="Indexing files"):

    file = f"embeddings_{i:04d}.pkl.pkl"
    path = os.path.join(input_dir, file)

    if not os.path.exists(path):
        print(f"File không tồn tại: {file}")
        continue

    with open(path, "rb") as f:
        data = pickle.load(f)

    total = len(data)
    success_count = 0
    fail_count = 0

    print(f"📄 File {file} có {total} vectors. Đang chia và upload theo lô {SUB_BATCH_SIZE}...")

    for start in tqdm(range(0, total, SUB_BATCH_SIZE), desc=f"→ File {file}", leave=False):
        end = min(start + SUB_BATCH_SIZE, total)
        sub_data = data[start:end]

        with collection.batch.dynamic() as batch:
            for obj in sub_data:
                try:
                    metadata = {
                        "page_content": obj["page_content"],
                        "pmid": obj["metadata"]["pmid"],
                        "title": obj["metadata"]["title"],
                        "abstract": obj["metadata"]["abstract"],
                        "chunk": obj["metadata"]["chunk"],
                    }

                    safe_uuid = generate_hashed_uuid(obj["vector"], obj["uuid"])

                    batch.add_object(
                        properties=metadata,
                        uuid=safe_uuid,
                        vector=obj["vector"]
                    )
                    success_count += 1

                except Exception as e:
                    print(f"Lỗi khi thêm object UUID={obj.get('uuid', '??')}: {e}")
                    fail_count += 1

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"{len(failed_objects)} lỗi trong batch {start}-{end}. Lỗi đầu tiên:")
            print(failed_objects[0])
            fail_count += len(failed_objects)
        else:
            print(f"Batch {start}-{end} OK")

        del sub_data
        gc.collect()

    del data
    gc.collect()
    print(f"Hoàn tất indexing file: {file}")
    print(f"Tổng vector trong file: {total}")
    print(f"Thành công: {success_count}")
    print(f"Thất bại: {fail_count}")
    print(f"Tổng đã xử lý: {success_count + fail_count}")

    # --- Restart container để giải phóng RAM ---
    print("Stop container để giải phóng RAM...")
    subprocess.run(["docker", "stop", container_name])
    time.sleep(60)  # Đợi RAM được hệ thống giải phóng (quan trọng)
    print("Start lại container...")
    subprocess.run(["docker", "start", container_name])
    print("✅ Container đã được khởi động lại.")

# --- Kết thúc ---
print("Đã hoàn tất indexing toàn bộ vectors vào Weaviate.")
client.close()
