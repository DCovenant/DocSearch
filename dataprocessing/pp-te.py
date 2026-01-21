#!/usr/bin/env python3
"""
Production OCR Text Extractor with Elasticsearch Integration
Optimized for high-throughput batch processing with tile splitting (1/2/4/8)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from paddleocr import PaddleOCR
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np
from numba import jit
import sys
import os
from pdf2image import convert_from_path
import argparse
from tqdm import tqdm
import gc
import cv2
from PIL import Image
import re

Image.MAX_IMAGE_PIXELS = None


# --- DATACLASSES ---

@dataclass
class OCRConfig:
    det_limit: int = 1920
    rec_model: str = "en_PP-OCRv5_mobile_rec"
    det_thresh: float = 0.3
    det_box_thresh: float = 0.6
    rec_batch_size: int = 16


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


# --- NUMBA ---

@jit(nopython=True, cache=True)
def normalize_coordinates_batch(coords: np.ndarray, dims: tuple[float, float]) -> np.ndarray:
    width, height = dims
    normalized = np.empty_like(coords)
    w_inv, h_inv = 1.0 / width, 1.0 / height
    for i in range(len(coords)):
        normalized[i, 0] = coords[i, 0] * w_inv
        normalized[i, 1] = coords[i, 1] * h_inv
        normalized[i, 2] = coords[i, 2] * w_inv
        normalized[i, 3] = coords[i, 3] * h_inv
    return normalized


# --- OCR ENGINE ---

class TextDetection:
    def __init__(self, config: OCRConfig) -> None:
        self.ocr = PaddleOCR(
            text_recognition_model_name=config.rec_model,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_det_limit_type='max',
            text_det_limit_side_len=config.det_limit,
            text_det_thresh=config.det_thresh,
            text_det_box_thresh=config.det_box_thresh,
            text_recognition_batch_size=config.rec_batch_size,
        )
        self.stats = {'files': 0, 'pages': 0, 'words': 0}

    def split_into_tiles(self, img: np.ndarray, config: TileConfig) -> list[TileData]:
        h, w = img.shape[:2]
        n, o = config.num_tiles, config.overlap
        
        if n == 1:
            return [TileData(img, 0, 0, 0)]
        
        if n == 2:
            mid_y = h // 2
            return [
                TileData(img[0:mid_y + o, :], 0, 0, 0),
                TileData(img[mid_y - o:h, :], 0, mid_y - o, 1)
            ]
        
        if n == 4:
            mid_y, mid_x = h // 2, w // 2
            return [
                TileData(img[0:mid_y + o, 0:mid_x + o], 0, 0, 0),
                TileData(img[0:mid_y + o, mid_x - o:w], mid_x - o, 0, 1),
                TileData(img[mid_y - o:h, 0:mid_x + o], 0, mid_y - o, 2),
                TileData(img[mid_y - o:h, mid_x - o:w], mid_x - o, mid_y - o, 3)
            ]
        
        if n == 8:
            # 4 columns x 2 rows
            q_x = w // 4
            mid_y = h // 2
            return [
                # Top row (4 tiles)
                TileData(img[0:mid_y + o, 0:q_x + o], 0, 0, 0),
                TileData(img[0:mid_y + o, q_x - o:2*q_x + o], q_x - o, 0, 1),
                TileData(img[0:mid_y + o, 2*q_x - o:3*q_x + o], 2*q_x - o, 0, 2),
                TileData(img[0:mid_y + o, 3*q_x - o:w], 3*q_x - o, 0, 3),
                # Bottom row (4 tiles)
                TileData(img[mid_y - o:h, 0:q_x + o], 0, mid_y - o, 4),
                TileData(img[mid_y - o:h, q_x - o:2*q_x + o], q_x - o, mid_y - o, 5),
                TileData(img[mid_y - o:h, 2*q_x - o:3*q_x + o], 2*q_x - o, mid_y - o, 6),
                TileData(img[mid_y - o:h, 3*q_x - o:w], 3*q_x - o, mid_y - o, 7)
            ]
        if n == 10:
            # 5 columns x 2 rows
            q_x = w // 5
            mid_y = h // 2
            return [
                # Top row (5 tiles)
                TileData(img[0:mid_y + o, 0:q_x + o], 0, 0, 0),
                TileData(img[0:mid_y + o, q_x - o:2*q_x + o], q_x - o, 0, 1),
                TileData(img[0:mid_y + o, 2*q_x - o:3*q_x + o], 2*q_x - o, 0, 2),
                TileData(img[0:mid_y + o, 3*q_x - o:4*q_x + o], 3*q_x - o, 0, 3),
                TileData(img[0:mid_y + o, 4*q_x - o:w], 4*q_x - o, 0, 4),
                # Bottom row (5 tiles)
                TileData(img[mid_y - o:h, 0:q_x + o], 0, mid_y - o, 5),
                TileData(img[mid_y - o:h, q_x - o:2*q_x + o], q_x - o, mid_y - o, 6),
                TileData(img[mid_y - o:h, 2*q_x - o:3*q_x + o], 2*q_x - o, mid_y - o, 7),
                TileData(img[mid_y - o:h, 3*q_x - o:4*q_x + o], 3*q_x - o, mid_y - o, 8),
                TileData(img[mid_y - o:h, 4*q_x - o:w], 4*q_x - o, mid_y - o, 9)
            ]

        raise ValueError(f"num_tiles must be 1, 2, 4, 8 or 10")

    def process_tile(self, td: TileData) -> tuple[Any, int, int, int]:
        result = self.ocr.predict(td.tile)
        return result, td.offset_x, td.offset_y, td.tile_idx

    def _in_overlap_region(self, center: tuple[float, float], td: TileData, config: TileConfig) -> bool:
        cx, cy = center
        n, o = config.num_tiles, config.overlap
        ox, oy, idx = td.offset_x, td.offset_y, td.tile_idx
        
        if n == 2:
            return idx == 1 and cy < oy + o
        
        if n == 4:
            if idx == 1: return cx < ox + o
            if idx == 2: return cy < oy + o
            if idx == 3: return cx < ox + o or cy < oy + o
        
        if n == 8:
            # Bottom row tiles check vertical overlap
            if idx >= 4 and cy < oy + o:
                return True
            # Non-leftmost tiles check horizontal overlap
            if idx in (1, 2, 3, 5, 6, 7) and cx < ox + o:
                return True
            
        if n == 10:
            # Bottom row tiles check vertical overlap
            if idx >= 5 and cy < oy + o:
                return True
            # Non-leftmost tiles check horizontal overlap
            if idx in (1, 2, 3, 4, 6, 7, 8, 9) and cx < ox + o:
                return True
        
        return False

    def merge_results(self, tile_results: list, config: TileConfig) -> list[WordResult]:
        merged = []
        for result, offset_x, offset_y, tile_idx in tile_results:
            if not result or not result[0]:
                continue
            td = TileData(None, offset_x, offset_y, tile_idx)
            for res in result:
                texts = res.rec_texts if hasattr(res, 'rec_texts') else res.get('rec_texts', [])
                scores = res.rec_scores if hasattr(res, 'rec_scores') else res.get('rec_scores', [])
                polys = res.rec_polys if hasattr(res, 'rec_polys') else res.get('rec_polys', [])
                for j, text in enumerate(texts):
                    if not text or not str(text).strip():
                        continue
                    poly = polys[j] if j < len(polys) else None
                    if poly is None:
                        continue
                    adjusted = [[float(pt[0]) + offset_x, float(pt[1]) + offset_y] for pt in poly]
                    center = (sum(pt[0] for pt in adjusted) / 4, sum(pt[1] for pt in adjusted) / 4)
                    if config.num_tiles > 1 and tile_idx > 0 and self._in_overlap_region(center, td, config):
                        continue
                    merged.append(WordResult(
                        text=str(text),
                        confidence=float(scores[j]) if j < len(scores) else 0.0,
                        box=adjusted,
                        tile_idx=tile_idx
                    ))
        return merged

    def process_pdf(self, file_path: str, config: ProcessConfig) -> list[PageResult]:
        images = convert_from_path(file_path, dpi=config.dpi)
        results = []
        
        for page_num, image in enumerate(images, 1):
            img = np.array(image)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            tiles = self.split_into_tiles(img_bgr, config.tiles)
            tile_results = [self.process_tile(td) for td in tiles]
            merged = self.merge_results(tile_results, config.tiles)
            
            self.stats['pages'] += 1
            self.stats['words'] += len(merged)
            
            results.append(PageResult(
                page_number=page_num,
                results=merged,
                width=img.shape[1],
                height=img.shape[0]
            ))
            
            del img, img_bgr, tiles, tile_results
            gc.collect()
        
        del images
        gc.collect()
        return results


# --- ELASTICSEARCH ---

def get_index_name(folder_path: str) -> str:
    return os.path.basename(os.path.normpath(folder_path)).lower().replace(' ', '_').replace('-', '_')


def reset_all_indices(es: Elasticsearch) -> int:
    """Delete all non-system indices"""
    indices = [i for i in es.indices.get_alias(index="*").keys() if not i.startswith('.')]
    for idx in indices:
        es.indices.delete(index=idx)
    return len(indices)

def setup_elasticsearch_index(es: Elasticsearch, index_name: str) -> None:
    """Create index if not exists, or recreate if exists"""
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
            "analysis": {"normalizer": {"lowercase_norm": {"type": "custom", "filter": ["lowercase"]}}}
        },
        "mappings": {
            "properties": {
                "folder_name": {"type": "keyword"},
                "file_name": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "source_folder": {"type": "keyword"},
                "image_info": {
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "dpi": {"type": "integer"},
                        "path": {"type": "keyword"}
                    }
                },
                "words": {
                    "type": "nested",
                    "properties": {
                        "word": {"type": "text", "fields": {"keyword": {"type": "keyword", "normalizer": "lowercase_norm", "ignore_above": 256}}},
                        "confidence": {"type": "float"},
                        "rotation": {"type": "integer"},
                        "coordinates": {"properties": {"x0": {"type": "float"}, "y0": {"type": "float"}, "x1": {"type": "float"}, "y1": {"type": "float"}}}
                    }
                }
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)


def bulk_index(es: Elasticsearch, pages: list[dict], timeout: int = 60) -> tuple[int, int]:
    if not pages:
        return 0, 0
    actions = [{'_index': p['_index'], '_id': p['_id'], '_source': p['_source']} for p in pages]
    success, failed = bulk(es.options(request_timeout=timeout), actions, chunk_size=500, raise_on_error=False)
    del actions
    gc.collect()
    return success, len(failed) if failed else 0


def to_es_format(file_path: str, page: PageResult, index_name: str, dpi: int, root_folder: str) -> dict:
    file_name = os.path.splitext(os.path.basename(file_path))[0].lower()
    # Relative path from root to file's parent (e.g., "folder1/folder2")
    rel_path = os.path.relpath(os.path.dirname(file_path), root_folder)
    source_folder = "" if rel_path == "." else rel_path

    word_docs = []
    valid = []
    coords = np.empty((len(page.results), 4), dtype=np.float32)

    for r in page.results:
        t = r.text.strip()
        if not t or re.fullmatch(r'[a-zA-Z]{1,2}|\d|\W+', t) or any(ord(c) > 127 for c in t):
            continue
        box = r.box
        coords[len(valid)] = [min(box[0][0], box[3][0]), min(box[0][1], box[1][1]), max(box[1][0], box[2][0]), max(box[2][1], box[3][1])]
        valid.append((t, r.confidence))

    if valid:
        coords = coords[:len(valid)]
        norm = normalize_coordinates_batch(coords, (float(page.width), float(page.height)))
        for i, (text, conf) in enumerate(valid):
            word_docs.append({
                "word": text,
                "confidence": float(conf),
                "rotation": 0,
                "coordinates": {"x0": float(norm[i, 0]), "y0": float(norm[i, 1]), "x1": float(norm[i, 2]), "y1": float(norm[i, 3])}
            })

    return {
        "_index": index_name,
        "_id": f"{file_name}_page_{page.page_number}",
        "_source": {
            "folder_name": index_name,
            "source_folder": source_folder,
            "file_name": file_name,
            "page_number": page.page_number,
            "image_info": {"width": page.width, "height": page.height, "dpi": dpi, "path": f"{file_name}_page_{page.page_number}.png"},
            "words": word_docs
        }
    }


# --- RENDER ---

def render_pdfs(folder_path: str, config: ProcessConfig) -> int:
    out = str(Path(__file__).resolve().parent.parent.joinpath('rendered_pages'))
    os.makedirs(out, exist_ok=True)
    
    pdfs = sorted([os.path.join(r, f) for r, _, fs in os.walk(folder_path) for f in fs if f.lower().endswith('.pdf') and not f.endswith('.Zone.Identifier')])
    if not pdfs:
        print(f"❌ No PDFs found in {folder_path}")
        return 0
    
    print(f"📁 Output: {out}\n📄 Found {len(pdfs)} PDFs")
    total = 0
    
    for pdf in tqdm(pdfs, desc="Rendering"):
        name = os.path.splitext(os.path.basename(pdf))[0].lower()
        for i, img in enumerate(convert_from_path(pdf, dpi=config.dpi), 1):
            img.save(os.path.join(out, f"{name}_page_{i}.png"), 'PNG')
            total += 1
        gc.collect()
    
    print(f"✅ Rendered {total} pages")
    return total


# --- CLI ---

def find_pdfs(folder: str) -> list[tuple[str, str]]:
    pdfs = []
    for root, _, files in os.walk(folder):
        src = os.path.basename(root) if root != folder else os.path.basename(folder)
        pdfs.extend((os.path.join(root, f), src) for f in files if f.lower().endswith('.pdf') and not f.endswith('.Zone.Identifier'))
    return sorted(pdfs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Production OCR with Elasticsearch")
    parser.add_argument("folder_path", nargs="?", help="Path to PDF folder")
    parser.add_argument("--tiles", type=int, choices=[1, 2, 4, 8, 10], default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--index", action="store_true", help="Index to Elasticsearch (appends to existing)")
    parser.add_argument("--reset", action="store_true", help="Reset ALL Elasticsearch indices")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    es_url = os.getenv('ES_URL', 'http://localhost:9200')
    es = Elasticsearch(es_url, request_timeout=60)

    # Handle --reset (can be standalone)
    if args.reset:
        if not es.ping():
            sys.exit(f"❌ Elasticsearch not reachable at {es_url}")
        count = reset_all_indices(es)
        print(f"✅ Reset {count} indices")
        if not args.folder_path:
            return

    if not args.folder_path:
        sys.exit("❌ folder_path required")

    if not os.path.isdir(args.folder_path):
        sys.exit(f"❌ Invalid directory: {args.folder_path}")

    if args.render_only and args.index:
        sys.exit("❌ --render-only and --index cannot be used together")

    proc_cfg = ProcessConfig(dpi=args.dpi, tiles=TileConfig(num_tiles=args.tiles, overlap=args.overlap))

    if args.render_only:
        render_pdfs(args.folder_path, proc_cfg)
        return

    pdfs = find_pdfs(args.folder_path)
    if not pdfs:
        sys.exit("❌ No PDFs found")

    print(f"📄 Found {len(pdfs)} PDFs")

    # Init OCR
    ocr = TextDetection(OCRConfig())

    # Init ES for indexing
    if args.index:
        if not es.ping():
            sys.exit(f"❌ Elasticsearch not reachable at {es_url}")
        index_name = get_index_name(args.folder_path)
        setup_elasticsearch_index(es, index_name)

    # Process
    stats = {'files': 0, 'pages': 0, 'words': 0, 'indexed': 0, 'failed': 0}
    pending: list[dict] = []
    index_name = get_index_name(args.folder_path) if args.index else None

    for idx, (pdf_path, _) in enumerate(tqdm(pdfs, desc="Processing"), 1):
        pages = ocr.process_pdf(pdf_path, proc_cfg)
        stats['files'] += 1

        for page in pages:
            stats['pages'] += 1
            stats['words'] += len(page.results)

            if args.index and page.results:
                pending.append(to_es_format(pdf_path, page, index_name, args.dpi, args.folder_path))

        if args.index and (len(pending) >= 500 or idx == len(pdfs)):
            s, f = bulk_index(es, pending)
            stats['indexed'] += s
            stats['failed'] += f
            pending.clear()
            if idx % 5 == 0:
                es.indices.refresh(index=index_name)

        if idx % 5 == 0:
            gc.collect()

    if args.index:
        es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "1s", "number_of_replicas": 1}})
        es.indices.refresh(index=index_name)

    print(f"\n{'='*60}\nFiles: {stats['files']}\nPages: {stats['pages']}\nWords: {stats['words']:,}")
    if args.index:
        print(f"Indexed: {stats['indexed']}")
    print(f"{'='*60}\n✅ Complete!")


if __name__ == "__main__":
    main()