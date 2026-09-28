import os, base64, io, re
from typing import Dict, List, Any
from PIL import Image
from openai import OpenAI
from agents.base_agent import BaseAgent

try:
    from cross_encoder_rerank import CrossEncoderReranker
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False

try:
    from ocr_search import OCRSearch
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from triple_filter import TripleFilter
    TRIPLE_AVAILABLE = True
except ImportError:
    TRIPLE_AVAILABLE = False

try:
    from sam_segmentation import SAMSegmentation
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    from multi_scale_clip import MultiScaleCLIPEncoder
    MSCLIP_AVAILABLE = True
except ImportError:
    MSCLIP_AVAILABLE = False

try:
    from adaptive_retrieval import AdaptiveRetrieval
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

try:
    from self_rag_filter import SelfRAGFilter
    SELFRAG_AVAILABLE = True
except ImportError:
    SELFRAG_AVAILABLE = False

try:
    from gradcam_attention import GradCAMAttention
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[GradCAM] gradcam_attention not available.")

class GLM4VAgentFinal(BaseAgent):
    API_KEY = os.environ.get("ZHIPU_API_KEY", "")
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    MODEL_NAME = "glm-4v-plus"
    MAX_TOKENS = 150
    TEMPERATURE = 0.1
    TOP_P = 0.9
    NUM_SEARCH_RESULTS = 5
    COARSE_TOPK = 20
    QUALITY_HARD_THRESHOLD = 0.25
    QUALITY_TOP1_CONFIDENT = 0.70
    W_CLIP = 0.6
    W_OCR = 0.3
    W_TRIPLE = 0.1
    USE_VLM_DESCRIPTION = True
    DESC_MAX_TOKENS = 30
    USE_MULTI_SCALE = True
    USE_SAM = True
    USE_GRADCAM = True
    USE_SELF_RAG = True
    USE_ADAPTIVE = True
    USE_RERANK = True
    USE_OCR = True
    USE_TRIPLE = True
    FEW_SHOT_EXAMPLES = (
        "\n[Examples]\n"
        "Q: Does this car have any electric variants?\nA: Yes, the Volkswagen Golf has electric variants.\n\n"
        "Q: How many calories are in this per serving?\nA: 470 calories.\n\n"
        "Q: Is this plant good for someone with asthma?\nA: Yes, it is good for asthma.\n"
    )
    hallucination_stats = {"intrinsic": 0, "extrinsic": 0, "confabulation": 0, "total": 0}

    def __init__(self, search_pipeline=None, kg_entities=None):
        super().__init__(search_pipeline)
        self.client = OpenAI(api_key=self.API_KEY, base_url=self.BASE_URL)
        self.kg_entities = kg_entities or []
        self.reranker = CrossEncoderReranker() if RERANK_AVAILABLE and self.USE_RERANK else None
        self.ocr_search = OCRSearch() if OCR_AVAILABLE and self.USE_OCR else None
        self.triple_filter = TripleFilter() if TRIPLE_AVAILABLE and self.USE_TRIPLE else None
        self.sam_seg = SAMSegmentation() if SAM_AVAILABLE and self.USE_SAM else None
        self.ms_encoder = MultiScaleCLIPEncoder() if MSCLIP_AVAILABLE and self.USE_MULTI_SCALE else None
        self.gradcam = GradCAMAttention() if GRADCAM_AVAILABLE and self.USE_GRADCAM else None
        self.adaptive = AdaptiveRetrieval() if ADAPTIVE_AVAILABLE and self.USE_ADAPTIVE else None
        self.self_rag = SelfRAGFilter() if SELFRAG_AVAILABLE and self.USE_SELF_RAG else None
        print(f"[GLM4VAgentFinal] Final integrated agent loaded")
        print(f"  Modules: rerank={self.reranker is not None}, ocr={self.ocr_search is not None}, triple={self.triple_filter is not None}, sam={self.sam_seg is not None}, gradcam={self.gradcam is not None}, msclip={self.ms_encoder is not None}, adaptive={self.adaptive is not None}, selfrag={self.self_rag is not None}")

    def get_batch_size(self):
        return 1

    def _encode_image(self, image):
        if image is None:
            return ""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.sam_seg is not None:
            try:
                image = self.sam_seg.segment_foreground(image)
            except Exception:
                pass
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _generate_image_description(self, image_b64):
        if not image_b64:
            return ""
        messages = [
            {"role": "system", "content": "Describe the image in one short sentence."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "What do you see? Answer in one sentence."}
            ]}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL_NAME, messages=messages,
                max_tokens=self.DESC_MAX_TOKENS, temperature=0.1)
            desc = response.choices[0].message.content.strip()
            print(f"[VLM Description] {desc}")
            return desc
        except Exception as e:
            print(f"[VLM Description] Failed: {e}")
            return ""

    def _retrieve_image_results(self, image, k=None):
        if image is None or self.search_pipeline is None:
            return []
        k = k or self.NUM_SEARCH_RESULTS
        try:
            results = self.search_pipeline(image, k=k)
            print(f"[Detection] Image Search: Called ({len(results) if results else 0} results)")
            candidates = []
            for r in results or []:
                if not isinstance(r, dict):
                    continue
                score = r.get("score", 0.0)
                for entity in r.get("entities", []):
                    if not isinstance(entity, dict):
                        continue
                    name = entity.get("entity_name", "")
                    attrs = entity.get("entity_attributes") or {}
                    if name:
                        candidates.append({"name": name, "attrs": attrs, "raw_score": score, "source": "clip"})
            return candidates
        except Exception as e:
            print(f"[Detection] Image Search: FAILED ({e})")
            return []

    def _retrieve_ocr_results(self, image):
        if self.ocr_search is None or image is None:
            return []
        try:
            import tempfile
            tmp_path = os.path.join(tempfile.gettempdir(), f"ocr_tmp_{os.getpid()}.jpg")
            image.save(tmp_path, "JPEG")
            results = self.ocr_search.search(tmp_path, self.kg_entities)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return results
        except Exception as e:
            print(f"[Detection] OCR Search: FAILED ({e})")
            return []

    def _retrieve_triple_results(self, query):
        if self.triple_filter is None or not self.kg_entities:
            return []
        try:
            results = self.triple_filter.pre_filter_pool(query, self.kg_entities)
            print(f"[Detection] Triple Filter: {len(results)} results")
            return results
        except Exception as e:
            print(f"[Detection] Triple Filter: FAILED ({e})")
            return []

    def _fuse_candidates(self, clip_cands, ocr_cands, triple_cands):
        fused_map = {}
        def add_to_map(cands, weight_key):
            for c in cands:
                name = c.get("name", "")
                if not name:
                    continue
                if name not in fused_map:
                    fused_map[name] = {"name": name, "attrs": c.get("attrs", {}), "clip_score": 0.0, "ocr_score": 0.0, "triple_score": 0.0}
                if weight_key == "clip":
                    fused_map[name]["clip_score"] = max(fused_map[name]["clip_score"], c.get("raw_score", 0.0))
                elif weight_key == "ocr":
                    fused_map[name]["ocr_score"] = max(fused_map[name]["ocr_score"], c.get("ocr_score", 0.0))
                elif weight_key == "triple":
                    fused_map[name]["triple_score"] = max(fused_map[name]["triple_score"], c.get("triple_score", 0.0))
        add_to_map(clip_cands, "clip")
        add_to_map(ocr_cands, "ocr")
        add_to_map(triple_cands, "triple")
        fused = []
        for name, data in fused_map.items():
            Q = self.W_CLIP * data["clip_score"] + self.W_OCR * data["ocr_score"] + self.W_TRIPLE * data["triple_score"]
            data["fused_score"] = Q
            fused.append(data)
        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused

    def _evaluate_candidate(self, candidate, query):
        attrs = candidate.get("attrs", {})
        raw_score = candidate.get("fused_score", candidate.get("raw_score", 0.0))
        relevance = raw_score
        num_attrs = len([v for v in attrs.values() if v and str(v).strip()])
        completeness = min(num_attrs / 10.0, 1.0)
        query_words = set(query.lower().split())
        match_count = 0
        for k, v in attrs.items():
            text = f"{k} {v}".lower()
            for qw in query_words:
                if qw in text:
                    match_count += 1
        attr_match = min(match_count / max(len(query_words) * 2, 1), 1.0)
        name = candidate.get("name", "").lower()
        name_match = 0.0
        if name:
            for qw in query_words:
                if len(qw) > 2 and qw in name:
                    name_match = 1.0
                    break
        quality = 0.5 * relevance + 0.1 * completeness + 0.2 * attr_match + 0.2 * name_match
        return {"score": quality, "relevance": relevance, "completeness": completeness, "attr_match": attr_match, "name_match": name_match}

    def _name_query_overlap(self, name, query):
        name_lower = name.lower()
        query_words = set(query.lower().split())
        stop_words = {"a","an","the","is","this","that","it","what","how","when","where","why","which","does","can","do","are","was","were","be","been","have","has","had","will","would","could","should","may","might","to","of","in","on","at","for","with","by","from","as","and","or","but","not","no","any","some","more","than","these","those","there","here","about","into","through","over"}
        keywords = {w for w in query_words if len(w) > 2 and w not in stop_words}
        for kw in keywords:
            if kw in name_lower:
                return True
        return False

    def _filter_and_select_strategy(self, candidates, query):
        if not candidates:
            return "drop", [], True
        scored = []
        for c in candidates:
            q = self._evaluate_candidate(c, query)
            scored.append((c, q))
        scored.sort(key=lambda x: x[1]["score"], reverse=True)
        print(f"[R2V-MIF Ranking] {len(scored)} candidates")
        for i, (c, q) in enumerate(scored[:5]):
            print(f"  #{i+1} {c.get('name','')[:30]} | Q={q['score']:.3f} (R={q['relevance']:.3f}, N={q['name_match']:.0f})")
        top_score = scored[0][1]["score"]
        if top_score < self.QUALITY_HARD_THRESHOLD:
            print(f"[Filtering] ALL DROPPED (top Q={top_score:.3f})")
            return "drop", [], True
        top1_name = scored[0][0].get("name", "")
        if top_score >= self.QUALITY_TOP1_CONFIDENT and self._name_query_overlap(top1_name, query):
            print(f"[Filtering] PRECISE mode (top-1 Q={top_score:.3f})")
            return "precise", [scored[0][0]], False
        kept = [c for c, q in scored if q["score"] >= self.QUALITY_HARD_THRESHOLD][:3]
        print(f"[Filtering] MULTI mode (kept {len(kept)} candidates)")
        return "multi", kept, False

    def _filter_attributes(self, entity, query):
        attrs = entity.get("attrs", {})
        if not attrs:
            return entity
        query_words = set(query.lower().split())
        attr_keywords = {
            "capacity": {"passenger","seat","capacity","carry","people"},
            "origin": {"origin","country","from","place","where","created","made"},
            "price": {"price","cost","cheap","expensive","$"},
            "time": {"year","built","construct","produce","when","time","first"},
            "size": {"size","dimension","length","width","height","weight"},
            "health": {"safe","danger","health","risk","harm","bad"},
            "performance": {"mpg","fuel","gas","mileage","engine","torque"},
            "location": {"location","city","state","country","address"},
        }
        matched_types = set()
        for qtype, keywords in attr_keywords.items():
            if any(kw in query_words for kw in keywords):
                matched_types.add(qtype)
        filtered = {}
        for k, v in attrs.items():
            if not v or not str(v).strip() or str(v).endswith(('.svg','.png','.jpg')):
                continue
            key_lower = k.lower()
            val_lower = str(v).lower()
            if any(qw in key_lower for qw in query_words):
                filtered[k] = v
                continue
            if any(qw in val_lower for qw in query_words):
                filtered[k] = v
                continue
            if matched_types:
                for qtype in matched_types:
                    if any(kw in key_lower for kw in attr_keywords[qtype]):
                        filtered[k] = v
                        break
        for i, (k, v) in enumerate(attrs.items()):
            if i >= 3:
                break
            if v and not str(v).endswith(('.svg','.png','.jpg')) and k not in filtered:
                filtered[k] = v
        return {**entity, "attrs": filtered}

    def _build_image_rag_context(self, strategy, candidates, query):
        if strategy == "precise" and candidates:
            c = self._filter_attributes(candidates[0], query)
            name = c.get("name", "")
            attrs = c.get("attrs", {})
            attr_items = []
            for k, v in attrs.items():
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                attr_items.append(f"{k}={v_str}")
            if attr_items:
                return f"[Image Search - Top Match] {name}\n{', '.join(attr_items)}"
            return f"[Image Search - Top Match] {name}"
        elif strategy == "multi" and candidates:
            lines = ["[Image Search - Candidates] The following entities were retrieved. Judge which one (if any) matches the image, then answer the question."]
            for i, c in enumerate(candidates, 1):
                c = self._filter_attributes(c, query)
                name = c.get("name", "")
                attrs = c.get("attrs", {})
                attr_summary = []
                for k, v in list(attrs.items())[:3]:
                    v_str = str(v)
                    if len(v_str) > 120:
                        v_str = v_str[:120] + "..."
                    attr_summary.append(f"{k}={v_str}")
                attr_str = " | ".join(attr_summary) if attr_summary else "(no key attributes)"
                lines.append(f"  {i}. {name}: {attr_str}")
            lines.append("If none match the image, answer based on your visual understanding.")
            return "\n".join(lines)
        return ""

    def _detect_hallucination(self, answer, strategy, candidates, rag_context):
        result = {"intrinsic": False, "extrinsic": False, "confabulation": False, "label": "none"}
        answer_lower = answer.lower()
        refusal_patterns = ["don't know","cannot answer","unable to","not sure","cannot determine","i don't know","no information"]
        is_refusal = any(p in answer_lower for p in refusal_patterns)
        certainty_words = [" is "," was "," are "," were "," has "," have "," founded in "," produced in "," located in "," manufactured in "]
        hedge_words = ["likely","probably","might be","could be","seems to be","appears to be","i think","i believe"]
        has_certainty = any(w in answer_lower for w in certainty_words)
        has_hedge = any(w in answer_lower for w in hedge_words)
        cand_info = [(c.get("name","").lower(), c.get("fused_score", c.get("raw_score", 0.0))) for c in candidates if c.get("name")]
        if strategy == "drop" and (not rag_context or not candidates):
            if not is_refusal and has_certainty and not has_hedge and len(answer) > 20:
                result["confabulation"] = True
                result["label"] = "confabulation"
                return result
        if candidates and strategy in ("precise","multi") and not is_refusal and cand_info:
            mentioned = [(name, score) for name, score in cand_info if name in answer_lower and len(name) > 2]
            if mentioned:
                cand_info_sorted = sorted(cand_info, key=lambda x: x[1], reverse=True)
                top_name, top_score = cand_info_sorted[0]
                for men_name, men_score in mentioned:
                    if men_name != top_name and top_score > men_score + 0.03:
                        result["intrinsic"] = True
                        result["label"] = "intrinsic"
                        return result
        if candidates and strategy in ("precise","multi") and not is_refusal and cand_info:
            has_any_name = any(name in answer_lower for name, _ in cand_info)
            if not has_any_name and has_certainty and not has_hedge:
                result["extrinsic"] = True
                result["label"] = "extrinsic"
                return result
        return result

    def _self_verify(self, answer, candidates, query):
        if not candidates or not answer:
            return answer
        return answer

    def _build_messages(self, query, image_b64, history, rag_context="", strategy="drop"):
        if strategy == "precise":
            system_prompt = "You are a helpful assistant with strong visual understanding.\nA high-confidence knowledge graph entity has been retrieved.\nStep 1 - Verify: Check if the entity matches what you see.\nStep 2 - Answer: Use the entity's attributes to answer concisely.\nIf it does not match, rely on your visual understanding.\nIf you truly cannot answer, say 'I don't know'.\nAnswer truthfully and concisely."
        elif strategy == "multi":
            system_prompt = "You are a helpful assistant with strong visual understanding.\nStep 1 - Describe: Look at the image and describe what you see in 1 sentence.\nStep 2 - Match: Compare your description to the candidate entities.\nStep 3 - Answer: Choose the best candidate and answer concisely.\nIf none match, answer based on your visual understanding.\nDo not say 'I don't know'. Answer truthfully and concisely."
        else:
            system_prompt = "You are a helpful assistant with strong visual understanding.\nNo relevant knowledge graph information was found.\nAnswer based solely on your visual understanding.\nIf you don't know, say 'I don't know'.\nAnswer truthfully and concisely."
        system_prompt += self.FEW_SHOT_EXAMPLES
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            recent = history[-4:] if len(history) > 4 else history
            for msg in recent:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    content = msg['content']
                    if isinstance(content, str) and len(content) > 500:
                        msg = {**msg, 'content': content[:500] + "..."}
                    messages.append(msg)
        user_content = []
        if image_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        text_parts = []
        if rag_context:
            text_parts.append(rag_context)
        text_parts.append(f"Question: {query}")
        if strategy == "precise":
            text_parts.append("Verify the retrieved entity against the image, then answer.")
        elif strategy == "multi":
            text_parts.append("Pick the matching candidate and answer, or use visual understanding.")
        else:
            text_parts.append("Answer based on what you see in the image.")
        user_content.append({"type": "text", "text": "\n\n".join(text_parts)})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _call_api(self, messages):
        response = self.client.chat.completions.create(
            model=self.MODEL_NAME, messages=messages,
            max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE, top_p=self.TOP_P)
        return response.choices[0].message.content.strip()

    def _apply_sam(self, image):
        if self.sam_seg is None or image is None:
            return image
        try:
            return self.sam_seg.segment_foreground(image)
        except Exception:
            return image

    def _run_retrieval_pipeline(self, image, query, use_ocr, use_triple, use_rerank, coarse_k):
        clip_cands = self._retrieve_image_results(image, k=coarse_k)
        ocr_cands = self._retrieve_ocr_results(image) if use_ocr else []
        triple_cands = self._retrieve_triple_results(query) if use_triple else []
        fused_cands = self._fuse_candidates(clip_cands, ocr_cands, triple_cands)
        if use_rerank and self.reranker is not None and len(fused_cands) > 3:
            fused_cands = self.reranker.rerank(query, fused_cands, top_k=10)
        if self.self_rag is not None and fused_cands:
            fused_cands = self.self_rag.score_candidates(query, fused_cands)
        strategy, kept_candidates, _ = self._filter_and_select_strategy(fused_cands, query)
        return strategy, kept_candidates, fused_cands

    def batch_generate_response(self, queries, images, message_histories):
        responses = []
        for query, image, history in zip(queries, images, message_histories):
            if self.adaptive is not None:
                config = self.adaptive.route(query)
                use_ocr = config.get("use_ocr", True)
                use_triple = config.get("use_triple", True)
                use_rerank = config.get("use_rerank", True)
                coarse_k = config.get("num_candidates", 20)
            else:
                use_ocr = self.USE_OCR
                use_triple = self.USE_TRIPLE
                use_rerank = self.USE_RERANK
                coarse_k = self.COARSE_TOPK
            strategy, kept_candidates, fused_cands = "drop", [], []
            working_image = image
            if working_image is not None:
                working_image = self._apply_sam(working_image)
                print("[CASCADE] Stage 1: SAM foreground segmentation")
                strategy, kept_candidates, fused_cands = self._run_retrieval_pipeline(working_image, query, use_ocr, use_triple, use_rerank, coarse_k)
            if strategy == "drop" and self.ms_encoder is not None and image is not None:
                print("[CASCADE] Stage 2: Multi-scale encoding (diagnostic)")
                try:
                    ms_feat = self.ms_encoder.encode_multi_scale(image)
                    print(f"[MultiScaleCLIP] Encoded, dim={ms_feat.shape}")
                except Exception as e:
                    print(f"[MultiScaleCLIP] Failed: {e}")
            if strategy == "drop" and self.gradcam is not None and image is not None:
                print("[CASCADE] Stage 3: Grad-CAM attention crop")
                try:
                    original_size = image.size
                    working_image = self.gradcam.crop_attention_region(image, query)
                    if working_image.size != original_size:
                        print(f"[GradCAM] Cropped attention region, size: {working_image.size}")
                    strategy, kept_candidates, fused_cands = self._run_retrieval_pipeline(working_image, query, use_ocr, use_triple, use_rerank, coarse_k)
                except Exception as e:
                    print(f"[GradCAM] Failed: {e}")
            image_b64 = self._encode_image(image) if image else ""
            rag_context = self._build_image_rag_context(strategy, kept_candidates, query)
            web_cands = []
            if self.USE_VLM_DESCRIPTION and image_b64:
                desc = self._generate_image_description(image_b64)
                if desc:
                    extended_query = f"{query} {desc}"
                    try:
                        web_cands = self._retrieve_web_results(extended_query)
                    except Exception:
                        pass
            if web_cands:
                web_ctx = self._build_web_rag_context(web_cands)
                if web_ctx:
                    rag_context = rag_context + "\n\n" + web_ctx if rag_context else web_ctx
            messages = self._build_messages(query, image_b64, history, rag_context, strategy)
            if image_b64:
                print(f"[Detection] Vision: Used (with {strategy} strategy)")
            else:
                print("[Detection] Vision: Not used")
            if rag_context:
                print(f"[RAG Context] {rag_context[:120]}...")
            else:
                print("[RAG Context] None")
            try:
                answer = self._call_api(messages)
                answer = self._self_verify(answer, kept_candidates, query)
                hallu = self._detect_hallucination(answer, strategy, kept_candidates, rag_context)
                self.hallucination_stats["total"] += 1
                for k in ("intrinsic","extrinsic","confabulation"):
                    if hallu.get(k):
                        self.hallucination_stats[k] += 1
                print(f"[Hallucination] label={hallu['label']} | stats={self.hallucination_stats}")
                responses.append(answer)
            except Exception as e:
                print(f"[API Error] {e}")
                responses.append("I don't know")
        return responses

    def _retrieve_web_results(self, query):
        if not query or self.search_pipeline is None:
            return []
        try:
            results = self.search_pipeline(query, k=self.NUM_SEARCH_RESULTS)
            return [r if isinstance(r, dict) else {"page_name": getattr(r, 'page_name', ''), "page_snippet": getattr(r, 'page_snippet', ''), "page_url": getattr(r, 'page_url', ''), "score": getattr(r, 'score', 0.0)} for r in results or []]
        except Exception as e:
            print(f"[Web Search] FAILED ({e})")
            return []

    def _build_web_rag_context(self, web_results):
        if not web_results:
            return ""
        lines = ["[Web Search Results]"]
        for r in web_results:
            if not isinstance(r, dict):
                continue
            name = r.get("page_name", "")
            snippet = str(r.get("page_snippet", ""))
            if name and snippet:
                lines.append(f"- {name}: {snippet[:250]}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def print_hallucination_report(self):
        total = self.hallucination_stats["total"]
        if total == 0:
            print("[Hallucination Report] No samples evaluated.")
            return
        print("\n" + "=" * 60)
        print("Hallucination Detection Report")
        print("=" * 60)
        print(f"Total samples : {total}")
        for k in ("confabulation","extrinsic","intrinsic"):
            v = self.hallucination_stats[k]
            print(f"{k:13s} : {v} ({v/total*100:.1f}%)")
        clean = total - sum(self.hallucination_stats[k] for k in ("intrinsic","extrinsic","confabulation"))
        print(f"{'Clean':13s} : {clean} ({clean/total*100:.1f}%)")
        print("=" * 60 + "\n")
