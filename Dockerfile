FROM paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9

WORKDIR /dataprocessing

RUN pip install --no-cache-dir \
    paddleocr \
    pdf2image \
    opencv-python-headless \
    numba \
    "elasticsearch>=8.0.0,<9.0.0" \
    tqdm \
    psutil

# Install poppler for pdf2image
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

ENV OUTPUT_DIR=/rendered_pages