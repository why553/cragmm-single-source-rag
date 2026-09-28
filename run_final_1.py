"""run_final_1.py - 单条快速验证"""
import os, sys, random
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
REPO_ROOT = r"C:\Users\proph\Desktop\暑期实训\CRAG-MM源码\meta-comprehensive-rag-benchmark-starter-kit-main"
PACK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PACK_ROOT, 'agents'))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('HF_TOKEN', '')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pyarrow.parquet as pq
from PIL import Image
import requests
from io import BytesIO
from cragmm_search.search import UnifiedSearchPipeline
from glm4v_agent_final import GLM4VAgentFinal

PARQUET_PATH = os.path.join(REPO_ROOT, 'data_cache', 'validation-00000-of-00005.parquet')
NUM_EXAMPLES, RANDOM_SEED = 1, 42

print("=" * 60 + "\nTask 1 - FINAL Single Example Quick Test\n" + "=" * 60)
print(f"\n[1] Loading {PARQUET_PATH}")
table = pq.read_table(PARQUET_PATH)
total_rows = len(table)
print(f"    Total rows: {total_rows}")
random.seed(RANDOM_SEED)
indices = random.sample(range(total_rows), min(NUM_EXAMPLES, total_rows))
print(f"    Selected: {len(indices)} example(s)")

print("\n[2] Initializing pipeline & agent...")
pipeline = UnifiedSearchPipeline(
    image_model_name='openai/clip-vit-large-patch14-336',
    image_hf_dataset_id='crag-mm-2025/image-search-index-validation',
    text_model_name='BAAI/bge-large-en-v1.5',
    web_hf_dataset_id='crag-mm-2025/web-search-index-validation',
    web_hf_dataset_tag=None,
    web_local_dir=os.path.join(REPO_ROOT, 'data_cache', 'web-search-index'),
)
agent = GLM4VAgentFinal(search_pipeline=pipeline, kg_entities=[])
print("    Agent ready")

print(f"\n[3] Running {len(indices)} example...\n" + "-" * 60)
for idx, i in enumerate(indices):
    turns = table['turns'][i].as_py()
    answers = table['answers'][i].as_py()
    image_url = table['image_url'][i].as_py()
    query = turns['query'][0] if turns['query'] else ""
    gt_answer = answers['ans_full'][0] if answers['ans_full'] else ""
    print(f"\n--- Example {idx+1}/{len(indices)} (row {i}) ---")
    print(f"Q: {query}\nGT: {gt_answer}")
    image = None
    try:
        img_dict = table['image'][i].as_py()
        if img_dict and isinstance(img_dict, dict) and 'bytes' in img_dict:
            image = Image.open(BytesIO(img_dict['bytes'])).convert('RGB')
        elif image_url and image_url != 'None':
            resp = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        print(f"Image failed: {e}")
    if image and query:
        try:
            responses = agent.batch_generate_response(queries=[query], images=[image], message_histories=[[]])
            pred = responses[0] if responses else "(no response)"
        except Exception as e:
            print(f"Gen error: {e}")
            pred = "(error)"
    else:
        pred = "(skipped)"
    print(f"Pred: {pred}\nMatch: {pred.lower().strip() == gt_answer.lower().strip()}")

agent.print_hallucination_report()
print("\n" + "=" * 60 + "\nSingle example test completed.\n" + "=" * 60)
