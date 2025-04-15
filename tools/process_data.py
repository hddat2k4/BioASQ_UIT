import gzip
import json
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


start = 1
end = 1274

def get_full_abstract(article_elem):
    abstract_elem = article_elem.find(".//Abstract")
    if abstract_elem is None:
        return None

    texts = []
    for abs_text in abstract_elem.findall("AbstractText"):
        label = abs_text.attrib.get("Label")
        part = abs_text.text or ""
        if label:
            texts.append(f"{label}: {part}")
        else:
            texts.append(part)
    return " ".join(texts).strip() if texts else None


def extract_all_pubmed_articles(gz_path, output_json_path, max_articles=None):
    articles = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag == "PubmedArticle":
                pmid = elem.findtext(".//PMID") or ""
                title = elem.findtext(".//ArticleTitle") or ""
                abstract = get_full_abstract(elem) or ""

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                })

                elem.clear()
                if max_articles and len(articles) >= max_articles:
                    break

    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(articles, out_f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu {len(articles)} bài báo vào {output_json_path}")

def extract(i):
    dir = "pubmed_baseline_2025"
    gz_file = f"pubmed25n{i:04d}.xml.gz"
    gz_path = os.path.join(dir, gz_file)

    if not os.path.exists(gz_path):
        print(f"⚠️ File {gz_file} không tồn tại, bỏ qua.")
        return

    output_dir = "pubmed_json_2025"
    os.makedirs(output_dir, exist_ok=True)
    json_file = f"pubmed25n{i:04d}.json"
    output_path = os.path.join(output_dir, json_file)

    extract_all_pubmed_articles(gz_path, output_path)

# Tiến trình xử lý nhiều file
for i in tqdm(range(start, end+1), desc="Đang xử lý các file PubMed"):
    extract(i)

