#!/usr/bin/env python3
"""
OCR Text Extractor with Elasticsearch Integration

# Extract OCR → JSONL
python3 pp-te.py folder1:index1 folder2:index2 --extract output.jsonl --dpi 200 --tiles 4

# Index JSONL → ES
python3 pp-te.py --index-jsonl output.jsonl

# Render pages only
python3 pp-te.py folder1 --render-only --dpi 300
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from paddleocr import PaddleOCR
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np
from numba import jit
import sys, os, gc, re, json, argparse
from pdf2image import convert_from_path
from tqdm import tqdm
import cv2
from PIL import Image, ImageSequence
import paddle
import psutil
from datetime import datetime
from pdf2image import pdfinfo_from_path
from pdf2image.exceptions import PDFPageCountError

Image.MAX_IMAGE_PIXELS = None


class MemoryLogger:
    def __init__(self, log_file: str = "metrics/memory_usage.log", interval: int = 1):
        self.log_file = log_file
        self.interval = interval
        self.page_count = 0
        with open(self.log_file, 'w') as f:
            f.write("timestamp,page,ram_used_gb,ram_percent,gpu_used_mb,gpu_total_mb,gpu_free_mb\n")

    def log(self):
        self.page_count += 1
        if self.page_count % self.interval != 0:
            return
        ram = psutil.virtual_memory()
        ram_gb = ram.used / (1024**3)
        try:
            gpu_used = paddle.device.cuda.memory_allocated() / (1024**2)
            gpu_total = paddle.device.cuda.max_memory_allocated() / (1024**2)
            gpu_free = (paddle.device.cuda.memory_reserved() - gpu_used) / (1024**2)  # Approximate free
        except Exception as e:
            gpu_used, gpu_total, gpu_free = 0, 0, 0
            print(f"GPU logging error: {e}")
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()},{self.page_count},{ram_gb:.2f},{ram.percent},{gpu_used:.0f},{gpu_total:.0f},{gpu_free:.0f}\n")


@dataclass
class TileConfig:
    num_tiles: int = 2
    overlap: int = 50


@dataclass
class ProcessConfig:
    dpi: int = 200
    tiles: TileConfig = None
    def __post_init__(self):
        if self.tiles is None:
            self.tiles = TileConfig()


@dataclass
class WordResult:
    text: str
    confidence: float
    box: list[list[float]]
    tile_idx: int


@dataclass
class PageResult:
    page_number: int
    results: list[WordResult]
    width: int
    height: int


@dataclass
class TileData:
    tile: np.ndarray
    offset_x: int
    offset_y: int
    tile_idx: int


@jit(nopython=True, cache=True)
def normalize_coordinates_batch(coords: np.ndarray, dims: tuple[float, float]) -> np.ndarray:
    w, h = dims
    out = np.empty_like(coords)
    for i in range(len(coords)):
        out[i, 0], out[i, 1] = coords[i, 0] / w, coords[i, 1] / h
        out[i, 2], out[i, 3] = coords[i, 2] / w, coords[i, 3] / h
    return out


class TextDetection:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_det_limit_type='max',
            text_det_limit_side_len=1920,
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_recognition_batch_size=16,
        )
        self.mem_logger = MemoryLogger()

    def split_into_tiles(self, img: np.ndarray, cfg: TileConfig) -> list[TileData]:
        h, w = img.shape[:2]
        n, o = cfg.num_tiles, cfg.overlap
        if n == 1:
            return [TileData(img, 0, 0, 0)]
        if n == 2:
            m = h // 2
            return [TileData(img[0:m+o, :], 0, 0, 0), TileData(img[m-o:h, :], 0, m-o, 1)]
        if n == 4:
            my, mx = h // 2, w // 2
            return [
                TileData(img[0:my+o, 0:mx+o], 0, 0, 0),
                TileData(img[0:my+o, mx-o:w], mx-o, 0, 1),
                TileData(img[my-o:h, 0:mx+o], 0, my-o, 2),
                TileData(img[my-o:h, mx-o:w], mx-o, my-o, 3)
            ]
        if n == 8:
            qx, my = w // 4, h // 2
            return [
                TileData(img[0:my+o, 0:qx+o], 0, 0, 0),
                TileData(img[0:my+o, qx-o:2*qx+o], qx-o, 0, 1),
                TileData(img[0:my+o, 2*qx-o:3*qx+o], 2*qx-o, 0, 2),
                TileData(img[0:my+o, 3*qx-o:w], 3*qx-o, 0, 3),
                TileData(img[my-o:h, 0:qx+o], 0, my-o, 4),
                TileData(img[my-o:h, qx-o:2*qx+o], qx-o, my-o, 5),
                TileData(img[my-o:h, 2*qx-o:3*qx+o], 2*qx-o, my-o, 6),
                TileData(img[my-o:h, 3*qx-o:w], 3*qx-o, my-o, 7)
            ]
        if n == 10:
            qx, my = w // 5, h // 2
            return [
                TileData(img[0:my+o, 0:qx+o], 0, 0, 0),
                TileData(img[0:my+o, qx-o:2*qx+o], qx-o, 0, 1),
                TileData(img[0:my+o, 2*qx-o:3*qx+o], 2*qx-o, 0, 2),
                TileData(img[0:my+o, 3*qx-o:4*qx+o], 3*qx-o, 0, 3),
                TileData(img[0:my+o, 4*qx-o:w], 4*qx-o, 0, 4),
                TileData(img[my-o:h, 0:qx+o], 0, my-o, 5),
                TileData(img[my-o:h, qx-o:2*qx+o], qx-o, my-o, 6),
                TileData(img[my-o:h, 2*qx-o:3*qx+o], 2*qx-o, my-o, 7),
                TileData(img[my-o:h, 3*qx-o:4*qx+o], 3*qx-o, my-o, 8),
                TileData(img[my-o:h, 4*qx-o:w], 4*qx-o, my-o, 9)
            ]
        raise ValueError(f"tiles must be 1,2,4,8,10")

    def _in_overlap(self, cx: float, cy: float, ox: int, oy: int, idx: int, cfg: TileConfig) -> bool:
        n, o = cfg.num_tiles, cfg.overlap
        if n == 2:
            return idx == 1 and cy < oy + o
        if n == 4:
            if idx == 1: return cx < ox + o
            if idx == 2: return cy < oy + o
            if idx == 3: return cx < ox + o or cy < oy + o
        if n == 8:
            if idx >= 4 and cy < oy + o: return True
            if idx in (1,2,3,5,6,7) and cx < ox + o: return True
        if n == 10:
            if idx >= 5 and cy < oy + o: return True
            if idx in (1,2,3,4,6,7,8,9) and cx < ox + o: return True
        return False

    def merge_results(self, tile_results: list, cfg: TileConfig) -> list[WordResult]:
        merged = []
        for result, ox, oy, idx in tile_results:
            if not result or not result[0]:
                continue
            for res in result:
                texts = res.rec_texts if hasattr(res, 'rec_texts') else res.get('rec_texts', [])
                scores = res.rec_scores if hasattr(res, 'rec_scores') else res.get('rec_scores', [])
                polys = res.rec_polys if hasattr(res, 'rec_polys') else res.get('rec_polys', [])
                for j, text in enumerate(texts):
                    if not text or not str(text).strip() or j >= len(polys) or polys[j] is None:
                        continue
                    adj = [[float(pt[0]) + ox, float(pt[1]) + oy] for pt in polys[j]]
                    cx, cy = sum(p[0] for p in adj) / 4, sum(p[1] for p in adj) / 4
                    if cfg.num_tiles > 1 and idx > 0 and self._in_overlap(cx, cy, ox, oy, idx, cfg):
                        continue
                    merged.append(WordResult(str(text), float(scores[j]) if j < len(scores) else 0.0, adj, idx))
        return merged

    def process_document(self, path: str, cfg: ProcessConfig) -> list[PageResult]:
        ext = os.path.splitext(path)[1].lower()
        results = []
        
        if ext == '.pdf':
            try:
                info = pdfinfo_from_path(path)
                total_pages = info['Pages']
            except (PDFPageCountError, ValueError):
                return []
            
            # Process in batches of 5 pages to balance speed and memory
            batch_size = 25
            for start_page in range(1, total_pages + 1, batch_size):
                end_page = min(start_page + batch_size - 1, total_pages)
                images = convert_from_path(path, dpi=cfg.dpi, first_page=start_page, last_page=end_page)
                
                for idx, image in enumerate(images):
                    page_num = start_page + idx
                    img = np.array(image)
                    image.close()
                    
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    h, w = img.shape[:2]
                    del img
                    
                    tiles = self.split_into_tiles(bgr, cfg.tiles)
                    del bgr
                    tile_res = []
                    for td in tiles:
                        tile_res.append((self.ocr.predict(td.tile), td.offset_x, td.offset_y, td.tile_idx))
                        del td.tile
                    del tiles
                    
                    merged = self.merge_results(tile_res, cfg.tiles)
                    del tile_res
                    results.append(PageResult(page_num, merged, w, h))
                    gc.collect()
                    paddle.device.cuda.empty_cache()
                    self.mem_logger.log()
                
                del images
                gc.collect()
        else:
            # TIFF processing
            tiff = Image.open(path)
            images = [f.convert('RGB') for f in ImageSequence.Iterator(tiff)]
            tiff.close()
            
            for page_num, image in enumerate(images, 1):
                img = np.array(image)
                image.close()
                del image
                
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]
                del img
                
                tiles = self.split_into_tiles(bgr, cfg.tiles)
                del bgr
                tile_res = []
                for td in tiles:
                    tile_res.append((self.ocr.predict(td.tile), td.offset_x, td.offset_y, td.tile_idx))
                    del td.tile
                del tiles
                
                merged = self.merge_results(tile_res, cfg.tiles)
                del tile_res
                results.append(PageResult(page_num, merged, w, h))
                gc.collect()
                paddle.device.cuda.empty_cache()
                self.mem_logger.log()
            del images
        
        gc.collect()
        return results


def sanitize_index_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def sanitize_filename(name: str) -> str:
    return re.sub(r'[#&?%\s\[\]{}|\\^~`]+', '_', name).strip('_')


def to_jsonl_record(path: str, page: PageResult, index_name: str, dpi: int, folder_path: str) -> dict:
    fname = sanitize_filename(os.path.splitext(os.path.basename(path))[0].lower())
    rel = os.path.relpath(os.path.dirname(path), folder_path)
    src = "" if rel == "." else rel
    words = []
    valid = []
    coords = np.empty((len(page.results), 4), dtype=np.float32)
    for r in page.results:
        t = r.text.strip()
        if not t or re.fullmatch(r'[a-zA-Z]{1,2}|\d|\W+', t) or any(ord(c) > 127 for c in t):
            continue
        b = r.box
        coords[len(valid)] = [min(b[0][0], b[3][0]), min(b[0][1], b[1][1]), max(b[1][0], b[2][0]), max(b[2][1], b[3][1])]
        valid.append((t, r.confidence))
    if valid:
        norm = normalize_coordinates_batch(coords[:len(valid)], (float(page.width), float(page.height)))
        for i, (t, c) in enumerate(valid):
            words.append({"word": t, "confidence": float(c), "rotation": 0,
                          "coordinates": {"x0": float(norm[i,0]), "y0": float(norm[i,1]), "x1": float(norm[i,2]), "y1": float(norm[i,3])}})
    return {
        "_index": index_name,
        "_id": f"{fname}_page_{page.page_number}",
        "_source": {
            "folder_path": folder_path, "folder_name": index_name, "source_folder": src,
            "file_name": fname, "page_number": page.page_number,
            "image_info": {"width": page.width, "height": page.height, "dpi": dpi, "path": f"{fname}_page_{page.page_number}.png"},
            "words": words
        }
    }


def get_processed_files(jsonl_path: str) -> set[str]:
    """Get set of already processed file_names from JSONL"""
    if not os.path.exists(jsonl_path):
        return set()
    processed = set()
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    processed.add(rec['_source']['file_name'])
    except Exception as e:
        print(f"⚠️ Error reading existing JSONL: {e}")
        return set()
    return processed


def find_documents(folder: str) -> list[str]:
    exts = ('.pdf', '.tiff', '.tif')
    return sorted([os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
                   if f.lower().endswith(exts) and not f.endswith('.Zone.Identifier')])


def extract_folder(folder: str, index_name: str, ocr: TextDetection, cfg: ProcessConfig, out_file: str) -> dict:
    docs = find_documents(folder)
    if not docs:
        print(f"⚠️ No documents in {folder}")
        return {'files': 0, 'pages': 0, 'words': 0}
    
    # Get already processed files to skip
    processed = get_processed_files(out_file)
    docs_to_process = [d for d in docs if sanitize_filename(os.path.splitext(os.path.basename(d))[0].lower()) not in processed]
    
    if not docs_to_process:
        print(f"✅ All {len(docs)} documents already processed in {folder}")
        return {'files': 0, 'pages': 0, 'words': 0}
    
    skipped = len(docs) - len(docs_to_process)
    print(f"\n📂 {folder} → {out_file} (index: {index_name}, {len(docs_to_process)} docs, {skipped} skipped)")
    stats = {'files': 0, 'pages': 0, 'words': 0}
    with open(out_file, 'a') as f:
        for doc in tqdm(docs_to_process, desc="Extracting"):
            pages = ocr.process_document(doc, cfg)
            stats['files'] += 1
            for p in pages:
                stats['pages'] += 1
                stats['words'] += len(p.results)
                if p.results:
                    f.write(json.dumps(to_jsonl_record(doc, p, index_name, cfg.dpi, folder)) + '\n')
            del pages
            gc.collect()
            paddle.device.cuda.empty_cache()
    return stats


def render_documents(folder: str, cfg: ProcessConfig) -> int:
    out = str(Path(__file__).resolve().parent.parent / 'rendered_pages')
    os.makedirs(out, exist_ok=True)
    docs = find_documents(folder)
    if not docs:
        print(f"❌ No documents in {folder}")
        return 0
    print(f"📁 Output: {out}\n📄 Found {len(docs)} documents")
    total = 0
    for doc in tqdm(docs, desc="Rendering"):
        name = sanitize_filename(os.path.splitext(os.path.basename(doc))[0].lower())
        ext = os.path.splitext(doc)[1].lower()
        if ext == '.pdf':
            images = convert_from_path(doc, dpi=cfg.dpi)
        else:
            tiff = Image.open(doc)
            images = [f.convert('RGB') for f in ImageSequence.Iterator(tiff)]
            tiff.close()
        for i, img in enumerate(images, 1):
            img.save(os.path.join(out, f"{name}_page_{i}.png"), 'PNG')
            total += 1
        gc.collect()
    print(f"✅ Rendered {total} pages")
    return total


def setup_es_index(es: Elasticsearch, name: str) -> None:
    if es.indices.exists(index=name):
        es.indices.delete(index=name)
    es.indices.create(index=name, body={
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1",
                     "analysis": {"normalizer": {"lowercase_norm": {"type": "custom", "filter": ["lowercase"]}}}},
        "mappings": {"properties": {
            "folder_path": {"type": "keyword"}, "folder_name": {"type": "keyword"},
            "file_name": {"type": "keyword"}, "page_number": {"type": "integer"}, "source_folder": {"type": "keyword"},
            "image_info": {"properties": {"width": {"type": "integer"}, "height": {"type": "integer"},
                                          "dpi": {"type": "integer"}, "path": {"type": "keyword"}}},
            "words": {"type": "nested", "properties": {
                "word": {"type": "text", "fields": {"keyword": {"type": "keyword", "normalizer": "lowercase_norm", "ignore_above": 256}}},
                "confidence": {"type": "float"}, "rotation": {"type": "integer"},
                "coordinates": {"properties": {"x0": {"type": "float"}, "y0": {"type": "float"},
                                               "x1": {"type": "float"}, "y1": {"type": "float"}}}
            }}
        }}
    })


def index_jsonl(path: str, es: Elasticsearch) -> None:
    if not os.path.exists(path):
        sys.exit(f"❌ File not found: {path}")
    indices, pending, stats = set(), [], {'indexed': 0, 'failed': 0}
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Indexing"):
            rec = json.loads(line)
            idx = rec['_index']
            if idx not in indices:
                setup_es_index(es, idx)
                indices.add(idx)
            pending.append({'_index': idx, '_id': rec['_id'], '_source': rec['_source']})
            if len(pending) >= 500:
                s, fail = bulk(es.options(request_timeout=60), pending, chunk_size=500, raise_on_error=False)
                stats['indexed'] += s
                stats['failed'] += len(fail) if fail else 0
                pending.clear()
    if pending:
        s, fail = bulk(es.options(request_timeout=60), pending, chunk_size=500, raise_on_error=False)
        stats['indexed'] += s
        stats['failed'] += len(fail) if fail else 0
    for idx in indices:
        es.indices.put_settings(index=idx, body={"index": {"refresh_interval": "1s", "number_of_replicas": 1}})
        es.indices.refresh(index=idx)
    print(f"✅ Indexed {stats['indexed']} docs to {len(indices)} indices")


def reset_all_indices(es: Elasticsearch) -> int:
    indices = [i for i in es.indices.get_alias(index="*").keys() if not i.startswith('.')]
    for idx in indices:
        es.indices.delete(index=idx)
    return len(indices)


def parse_folders(args: list[str]) -> list[tuple[str, str]]:
    result = []
    for item in args:
        if ':' in item:
            folder, name = item.rsplit(':', 1)
        else:
            folder, name = item, os.path.basename(os.path.normpath(item))
        result.append((folder, sanitize_index_name(name)))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="OCR Extractor", epilog="Format: folder:indexname or folder")
    p.add_argument("folders", nargs="*", help="folder:indexname pairs")
    p.add_argument("--tiles", type=int, choices=[1, 2, 4, 8, 10], default=2)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--extract", metavar="FILE", help="Extract OCR to JSONL")
    p.add_argument("--index-jsonl", metavar="FILE", help="Index JSONL to ES")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--reset", action="store_true", help="Reset all ES indices")
    args = p.parse_args()

    es_url = os.getenv('ES_URL', 'http://localhost:9200')
    es = Elasticsearch(es_url, request_timeout=60)

    if args.reset:
        if not es.ping():
            sys.exit(f"❌ ES not reachable at {es_url}")
        print(f"✅ Reset {reset_all_indices(es)} indices")
        if not args.folders and not args.index_jsonl:
            return

    if args.index_jsonl:
        if not es.ping():
            sys.exit(f"❌ ES not reachable at {es_url}")
        index_jsonl(args.index_jsonl, es)
        return

    if not args.folders:
        sys.exit("❌ At least one folder required")

    pairs = parse_folders(args.folders)
    for folder, _ in pairs:
        if not os.path.isdir(folder):
            sys.exit(f"❌ Invalid directory: {folder}")

    cfg = ProcessConfig(dpi=args.dpi, tiles=TileConfig(num_tiles=args.tiles, overlap=args.overlap))

    if args.render_only:
        for folder, _ in pairs:
            render_documents(folder, cfg)
        return

    if not args.extract:
        sys.exit("❌ --extract FILE required for OCR extraction")

    ocr = TextDetection()
    totals = {'files': 0, 'pages': 0, 'words': 0}
    # Don't clear file if it exists (resume mode)
    if not os.path.exists(args.extract):
        open(args.extract, 'w').close()
    else:
        print(f"📄 Resuming: {args.extract} already exists, will skip processed files")
    for folder, idx_name in pairs:
        stats = extract_folder(folder, idx_name, ocr, cfg, args.extract)
        for k in totals:
            totals[k] += stats[k]
    print(f"\n{'='*60}\nFiles: {totals['files']} | Pages: {totals['pages']} | Words: {totals['words']:,}\nOutput: {args.extract}\n{'='*60}")


if __name__ == "__main__":
    main()
