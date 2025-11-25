
import json
from pathlib import Path
import pyarrow.parquet as pq

try:
    first_parquet_file = next((Path("/Users/macbookpro/Desktop/workspace/handcap/data/peg_in_hole_combined/data").rglob("*.parquet")))
    print(f"正在检查第一个 Parquet 文件: {first_parquet_file}")
    meta = pq.read_schema(first_parquet_file).metadata
    if meta and b"huggingface" in meta:
        hf_meta = json.loads(meta[b"huggingface"])
        print("Hugging Face 元数据中的特征:")
        print(json.dumps(hf_meta["info"]["features"], indent=2))
except Exception as e:
    print(f"读取 Parquet 元数据时出错: {e}")