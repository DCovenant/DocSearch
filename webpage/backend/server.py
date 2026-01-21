from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query, Path as FastAPIPath
from pydantic import BaseModel
from elasticsearch import AsyncElasticsearch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
from PIL import Image

# --- ELASTICSEARCH CONFIG ---
ES_URL = os.getenv("ES_URL", "http://localhost:9200")  # Use env var or default to localhost
ALL_INDICES = "*"  # Wildcard to search all indices
PRIMARY_IMAGES_FOLDER = "images"  # Folder where the unrendered images are
RENDERED_IMAGES_FOLDER = "rendered_pages"

# Global Elasticsearch client for reuse across requests
es_client: AsyncElasticsearch | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Starts and stops async elasticsearch instance
    """
    global es_client # Tells python to modify the global variable and not create a local one
    
    try:
        es_client = AsyncElasticsearch([ES_URL], request_timeout=30)    # Startup: Create persistent Elasticsearch client
    except Exception as e:
        raise HTTPException(status_code=502, detail={"message":"Elasticsearch error","error": str(e)})

    yield  # Application is running
    
    # Shutdown: Close Elasticsearch client
    if es_client:
        await es_client.close()
        es_client = None

app = FastAPI(lifespan=lifespan)

# Enable CORS (allow all origins for development)
# When the Vue dev server runs (default port 5173 for Vite) it is a different origin than
# this FastAPI server (localhost:3000). Browsers block requests across origins unless CORS
# headers are present. This middleware sets permissive headers so the frontend can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # change to ["*"] for quick dev testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all request headers
)

# Path to rendered_pages
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
rendered_pages_path = os.path.join(project_root, 'rendered_pages')

if os.path.exists(rendered_pages_path):
    # Create a custom StaticFiles class with CORS headers
    class CORSStaticFiles(StaticFiles):
        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await super().__call__(scope, receive, send)
            
            # For non-http requests (like websocket), we need to create a new send function
            # that adds CORS headers to the response
            async def send_with_cors(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Add CORS headers
                    headers[b"access-control-allow-origin"] = b"http://localhost:5173"
                    headers[b"access-control-allow-methods"] = b"GET, HEAD, OPTIONS"
                    headers[b"access-control-allow-headers"] = b"*"
                    
                    message["headers"] = [(k, v) for k, v in headers.items()]
                
                await send(message)
            
            return await super().__call__(scope, receive, send_with_cors)

    app.mount("/renderedpages", CORSStaticFiles(directory=rendered_pages_path), name="rendered_pages")
    print(f"Mounted rendered_pages directory with CORS headers: {rendered_pages_path}")
else:
    print(f"Warning: rendered_pages directory not found at {rendered_pages_path}")

# Model for POST body validation
# Pydantic model describing the expected JSON body for the /search POST route.
# When FastAPI sees this type used as a parameter it will automatically parse and validate
# the incoming JSON against this model and return a 422 error if the shape/type is wrong.
class WordCheckRequest(BaseModel):
    searchTerm: str | int
    highlightAllWords: bool = False
    exactMatch: bool = False
    indexName: str = "*"  # Target index, defaults to all

def generate_prefix_queries(term: str, base_boost: float = 10.0) -> list[dict]:
    """Generate progressively shorter prefix queries with decreasing boost"""
    words = term.split()
    queries = []
    # Full term first
    queries.append({"term": {"words.word.keyword": {"value": term, "boost": base_boost}}})

    if len(words) > 1:
        # Progressive word removal from end: "red phase appears" -> "red phase" -> "red"
        for i in range(len(words) - 1, 0, -1):
            partial = " ".join(words[:i])
            boost = base_boost * (i / len(words)) * 0.8
            queries.append({"term": {"words.word.keyword": {"value": partial, "boost": boost}}})

    # Progressive char removal from last word: "red phase" -> "red phas" -> "red pha"...
    if words:
        last_word = words[-1]
        prefix = " ".join(words[:-1]) + " " if len(words) > 1 else ""
        for i in range(len(last_word) - 1, 0, -1):
            partial = (prefix + last_word[:i]).strip()
            boost = base_boost * (i / len(last_word)) * 0.5
            queries.append({"term": {"words.word.keyword": {"value": partial, "boost": boost}}})

    return queries

# --- ROUTES ---

@app.get("/indices")
async def list_indices():
    """List all non-system Elasticsearch indices with doc counts"""
    try:
        stats = await es_client.cat.indices(format="json")
        indices = [
            {"name": idx["index"], "docs": int(idx.get("docs.count", 0))}
            for idx in stats if not idx["index"].startswith(".")
        ]
        return {"indices": sorted(indices, key=lambda x: x["name"])}
    except Exception as e:
        raise HTTPException(status_code=502, detail={"message": "Elasticsearch error", "error": str(e)})

@app.get("/suggest")
async def suggest(q: str = Query(..., min_length=1)):
    """
    Suggest words using Elasticsearch prefix query on nested words.
    Returns unique words that start with the given prefix.
    """
    # Use prefix query on nested words.word.keyword field
    suggest_body = {
        "size": 0,  # We don't need the full documents
        "query": {
            "nested": {
                "path": "words",
                "query": {
                    "prefix": {
                        "words.word.keyword": {
                            "value": q.lower(),
                            "case_insensitive": True
                        }
                    }
                }
            }
        },
        "aggs": {
            "words_agg": {
                "nested": {
                    "path": "words"
                },
                "aggs": {
                    "filtered_words": {
                        "filter": {
                            "prefix": {
                                "words.word.keyword": {
                                    "value": q.lower(),
                                    "case_insensitive": True
                                }
                            }
                        },
                        "aggs": {
                            "unique_words": {
                                "terms": {
                                    "field": "words.word.keyword",
                                    "size": 100,
                                    "order": {"_key": "asc"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    try:
        resp = await es_client.search(index=ALL_INDICES, body=suggest_body)
        
        suggestions = []
        aggs = resp.get("aggregations", {})
        words_agg = aggs.get("words_agg", {})
        filtered = words_agg.get("filtered_words", {})
        buckets = filtered.get("unique_words", {}).get("buckets", [])
        
        for bucket in buckets:
            word = bucket.get("key", "")
            if word:
                suggestions.append(word)
        
        return {"suggestions": suggestions}
    
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Elasticsearch error",
                "error": str(e),
                "query": suggest_body
            }
        )

# POST search route with validation
@app.post("/wordcheck")
async def wordcheck(body: WordCheckRequest):
    """
    This is the main route the Vue frontend uses.

    Expected input (JSON body): { "searchTerm": "text to search" }

    Workflow:
    1. Validate and sanitize the input (Pydantic already ensures `searchTerm` is present and a string).
    2. Expects the word mappings to be nested.
    3. Build an appropriate Elasticsearch query that searches `words.word` and also has a
       fallback search across other fields.
    4. Send the query to Elasticsearch and parse the response. If matches are found,
       inspect `_source.words` to locate the actual OCR token objects and extract coordinates
       to return to the frontend.

    The Vue frontend calls this route (POST) and expects a response shaped like:
    { success: true|false, message: "...", results: { fileName, pageNumber, totalHits, matches, ... } }
    """
    # Extract the validated search term from the Pydantic model instance.
    searchTerm = body.searchTerm
    highlight_all = body.highlightAllWords
    exactMatch = body.exactMatch
    target_index = body.indexName if body.indexName else ALL_INDICES

    # Input validation
    if not searchTerm or searchTerm.strip() == "":
        raise HTTPException(status_code=400, detail="Search term cannot be empty")

    if len(searchTerm) >= 100:
        raise HTTPException(status_code=400, detail="Search term too long")

    # Sanitize input
    cleanSearchTerm = searchTerm.strip().replace('<', '').replace('>', '').replace('"', '').replace("'", '').replace('&', '')
    if cleanSearchTerm == "":
        raise HTTPException(status_code=400, detail="Search term contains only invalid characters")

    # MODE 1: Highlight all OCR words (search by filename)
    if highlight_all:
        # Build query to match by filename (without extension)
        searchBody = {
            "query": {
                "bool": {
                    "should": [
                        # Exact filename match
                        {"term": {"file_name.keyword": {"value": cleanSearchTerm, "boost": 10.0}}},
                        # Partial filename match
                        {"wildcard": {"file_name": {"value": f"*{cleanSearchTerm}*", "boost": 5.0}}},
                        # Match with .pdf extension
                        {"term": {"file_name.keyword": {"value": f"{cleanSearchTerm}.pdf", "boost": 8.0}}}
                    ]
                }
            },
            "size": 100
        }
        
        try:
            resp = await es_client.search(index=target_index, body=searchBody)
            hits = resp.get("hits", {}).get("hits", [])

            if not hits:
                return {
                    "success": False,
                    "message": f"File '{cleanSearchTerm}' not found.",
                    "results": None
                }
            
            # Process all matching files/pages
            all_results = []
            for hit in hits:
                source = hit.get("_source", {})
                indexName = hit["_index"]
                pageNumber = source.get("page_number")
                folderName = source.get("folder_name", "unknown")
                fileName = source.get("file_name", indexName)
                
                # Get ALL words from the OCR data
                all_words = source.get("words", [])
                
                # Convert all words to match format with full confidence (1.0)
                page_results = [
                    {
                        "word": w.get("word"),
                        "coordinates": w.get("coordinates"),
                        "score": 1.0  # Full confidence for all OCR words
                    }
                    for w in all_words if w.get("word") and w.get("coordinates")
                ]
                
                if page_results:
                    all_results.append({
                        "folderName": folderName,
                        "fileName": fileName,
                        "pageNumber": pageNumber,
                        "matches": page_results
                    })
            
            match_count = sum(len(r["matches"]) for r in all_results)
            page_count = len(all_results)
            
            return {
                "success": True,
                "message": f"Found {match_count} OCR words across {page_count} page(s) in '{cleanSearchTerm}'",
                "results": {
                    "fileName": all_results[0]["fileName"] if all_results else "",
                    "pageNumber": all_results[0]["pageNumber"] if all_results else 0,
                    "totalHits": page_count,
                    "matches": all_results[0]["matches"] if all_results else [],
                    "allMatches": all_results,
                }
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Elasticsearch error",
                    "error": str(e),
                    "query": searchBody
                }
            )
    
    # MODE 2: Normal word search (existing code)

    # Build Elasticsearch query with exact match first (fast term query on normalized keyword)
    # then fallback to phrase match (full-text). Use inner_hits to return only matching nested objects.
    if exactMatch:
        # Exact match query - only exact word matches
        words_clause = {
            "nested": {
                "path": "words",
                "query": {
                    "term": {
                        "words.word.keyword": {
                            "value": cleanSearchTerm
                        }
                    }
                },
                "inner_hits": {
                    "name": "matched_words", 
                    "size": 100, 
                    "_source": True
                }
            }
        }
    else:
        # Generate progressive prefix queries for split-word matching
        prefix_queries = generate_prefix_queries(cleanSearchTerm.lower())

        words_clause = {
            "nested": {
                "path": "words",
                "query": {
                    "bool": {
                        "should": prefix_queries + [
                            {"match_phrase": {"words.word": {"query": cleanSearchTerm, "slop": 0, "boost": 5.0}}},
                            {"match": {"words.word": {"query": cleanSearchTerm, "boost": 2.0}}},
                            {"wildcard": {"words.word": {"value": f"*{cleanSearchTerm}*", "boost": 3.0}}},
                            {"fuzzy": {"words.word": {"value": cleanSearchTerm, "fuzziness": "AUTO", "boost": 1.5}}}
                        ]
                    }
                },
                "inner_hits": {
                    "name": "matched_words",
                    "size": 100,
                    "_source": True
                }
            }
        }

    # Compose the final query body
    searchBody = {
        "query": {
            "bool": {
                "should": [
                    words_clause,
                    {
                        "simple_query_string": {
                            "query": cleanSearchTerm
                        }
                    }
                ]
            }
        },
        "size": 100
    }

    try:
        # Use the official Elasticsearch client's search method
        resp = await es_client.search(index=target_index, body=searchBody)
        
        # Parse the response (no need to call .json(), client returns dict)
        hits = resp.get("hits", {})
        total_raw = hits.get("total", 0)
        
        # Normalize total to an integer
        if isinstance(total_raw, dict):
            totalHits = total_raw.get("value", 0)
        else:
            try:
                totalHits = int(total_raw)
            except Exception:
                totalHits = 0

        if totalHits == 0:
            return {
                "success": False,
                "message": "Word not found.",
                "results": None
            }

        # Process ALL results using inner_hits for matched nested objects
        all_results = []
        all_hits = hits.get("hits", [])
        
        if all_hits:
            # Get first hit info for backward compatibility
            firstHit = all_hits[0]
            first_indexName = firstHit["_index"]
            first_source = firstHit.get("_source", {})
            first_pageNumber = first_source.get("page_number")
            
            # Process each hit to build a list of all matches across all pages/files
            for hit in all_hits:
                indexName = hit["_index"]
                source = hit.get("_source", {})
                pageNumber = source.get("page_number")
                
                # Extract matches from inner_hits (more efficient than scanning _source)
                inner_hits = hit.get("inner_hits", {})
                matched_words_hits = inner_hits.get("matched_words", {}).get("hits", {}).get("hits", [])
                
                page_results = []
                
                if matched_words_hits:
                    # Use inner_hits results
                    for nested_hit in matched_words_hits:
                        nested_src = nested_hit.get("_source", {})
                        # Get score from the hit
                        match_score = nested_hit.get("_score", 1.0)
                        # Normalize score to a 0-1 range (assuming typical Elasticsearch scores)
                        # We'll use a simple normalization approach based on common ES score ranges
                        normalized_score = min(match_score / 15.0, 1.0)
                        
                        page_results.append({
                            "word": nested_src.get("word"),
                            "coordinates": nested_src.get("coordinates"),
                            "score": normalized_score
                        })
                else:
                    # Fallback: scan _source if inner_hits is empty
                    for w in source.get("words", []):
                        w_text = w.get("word")
                        if not w_text:
                            continue
                        if w_text.lower() == cleanSearchTerm.lower() or cleanSearchTerm.lower() in w_text.lower():
                            # Calculate a simple confidence score based on exact match vs partial match
                            confidence = 0.9 if w_text.lower() == cleanSearchTerm.lower() else 0.5
                            
                            page_results.append({
                                "word": w_text,
                                "coordinates": w.get("coordinates"),
                                "score": confidence
                            })
                
                # Add this page's results to the all_results list
                if page_results:
                    # Get folder name and file name from the source
                    folderName = source.get("folder_name", "unknown")
                    fileName = source.get("file_name", indexName)
                    sourcefolder = source.get("source_folder","unknown")
                    
                    all_results.append({
                        "folderName": folderName,
                        "sourcefolder": sourcefolder,
                        "fileName": fileName,
                        "pageNumber": pageNumber,
                        "matches": page_results
                    })
            
            # Construct message with total matches across all pages
            match_count = len(all_results)
            message_text = f"{cleanSearchTerm} found in {match_count} " + ("page" if match_count == 1 else "pages")
            
            return {
                "success": True,
                "message": message_text,
                "results": {
                    "fileName": first_indexName,  # Keep first result for backward compatibility
                    "pageNumber": first_pageNumber,
                    "totalHits": totalHits,
                    "matches": all_results[0]["matches"] if all_results else [],  # First page matches for backward compatibility
                    "allMatches": all_results,  # All matches across all pages/files
                }
            }
        else:
            return {
                "success": False,
                "message": "Word not found.",
                "results": None
            }
            
    except Exception as e:
        # Elasticsearch client provides detailed error information
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Elasticsearch error",
                "error": str(e),
                "query": searchBody
            }
        )

# Add a direct path to access pre-rendered images
@app.get("/rendered-image/{file_name}")
async def get_rendered_image(file_name: str = FastAPIPath(..., description="Image filename with extension")):
    """
    Direct access to pre-rendered images by filename.
    
    This route provides a simple way to access pre-rendered images created by the OCR process.
    Example: /rendered-image/246_1104_1159_01_page_1.png
    """
    try:
        # Path to the rendered image
        image_path = os.path.join(rendered_pages_path, file_name)
        
        # Check if the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image {file_name} not found")
            
        # Get image dimensions for headers
        with Image.open(image_path) as img:
            width, height = img.size
            
        # Return the image
        return FileResponse(
            path=image_path,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename={file_name}",
                "X-Page-Width": str(width),
                "X-Page-Height": str(height),
                "X-Page-DPI": "200"  # Standard OCR DPI
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

# --- To run the server, use this command in your terminal ---
# uvicorn server:app --reload --port 3000
