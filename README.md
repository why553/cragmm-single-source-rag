# Task 1 说明

## 一、说明：未打包大型依赖库

本目录仅包含**核心代码**，未打包以下大型第三方库与预训练模型如下：

| 依赖项 | 大小 | 说明 |
|---|---|---|
| PyTorch | ~2GB | 深度学习框架，由 pip/conda 统一安装 |
| Transformers | ~500MB | HuggingFace 模型库，运行时自动下载缓存 |
| CLIP (ViT-B/32, ViT-L/14) | ~350MB~1.5GB | 预训练视觉-语言模型，首次运行自动下载 |
| SAM (ViT-B) | ~375MB | 前景分割预训练权重，已提供下载脚本 |
| Cross-encoder (MiniLM-L6) | ~80MB | 精排模型，首次运行自动下载 |
| PaddleOCR 模型 | ~30MB | OCR 推理模型，首次运行自动下载 |
| ChromaDB / 向量索引 | ~5GB | 数据集索引文件，由比赛方提供，不在代码包内 |

**运行前请确保安装：**
```bash
pip install openai pillow torch transformers sentence-transformers paddleocr thefuzz jieba segment-anything pyarrow requests
```

---

## 二、文件清单

### 主程序

| 文件 | 说明 |
|---|---|
| `run_final_1.py` | 单条样本快速验证脚本，确认全部模块加载与流程通畅。 |
| `run_final_100.py` | 100 条随机样本端到端评测脚本，输出 Exact Match 与幻觉统计。 |

### Agent 核心模块（`agents/`）

| 文件 | 说明 |
|---|---|
| `base_agent.py` | 定义 Agent 基类与检索管道接口，供主 Agent 继承。 |
| `glm4v_agent_final.py` | **主入口**：集成全部策略的 GLM-4V Agent，负责多路检索、融合、过滤、Prompt 构建、API 调用与幻觉检测。 |
| `cross_encoder_rerank.py` | 二阶段精排器，用 `ms-marco-MiniLM-L-6-v2` 对 query-候选交叉编码，输出精细相关性分数。 |
| `ocr_search.py` | OCR 辅助检索器，调用 PaddleOCR 提取图中可见文字，与 KG 实体名做模糊匹配。 |
| `triple_filter.py` | 三元组关键词预筛器，用 jieba 分词提取 query 关键词，在 mock KG 中做属性值过滤。 |
| `sam_segmentation.py` | SAM 前景分割器，自动分割图像主体物体并去除背景噪声，提升 CLIP 检索精度。 |
| `multi_scale_clip.py` | 多尺度 CLIP 编码器，同时编码原图、局部裁剪、上采样三路特征并拼接，捕获小目标细节。 |
| `gradcam_attention.py` | 区域注意力裁剪器，基于 CLIP attention map 定位与 query 最相关的图像区域并裁剪。 |
| `adaptive_retrieval.py` | 自适应检索路由器，根据问题复杂度自动选择轻量/完整检索策略，节省 API 开销。 |
| `self_rag_filter.py` | Self-RAG 候选过滤器，调用大模型对每个检索候选独立打分，低分候选自动丢弃。 |

### 模型权重（`checkpoints/`）

| 文件 | 说明 |
|---|---|
| `sam_vit_b_01ec64.pth` | SAM 前景分割预训练权重（ViT-B，375MB），已从 Git 中排除，请从 [Segment Anything 官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints) 下载后放入此目录。 |

---

## 三、环境变量

运行前需设置以下密钥（代码中不包含任何明文密钥）：

```bash
# Windows PowerShell
$env:ZHIPU_API_KEY = "你的智谱API Key"
$env:HF_TOKEN = "你的HuggingFace Token"   # 可选，国内镜像一般不需要
```

---

## 四、快速运行

```bash
cd single_source
py -3.13 run_final_1.py     # 单条验证
py -3.13 run_final_100.py   # 100条评测
```
