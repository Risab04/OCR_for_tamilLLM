# OCR_for_tamilLLM
I have evaluated multiple OCR for tamil text extraction for training tamil LLM

OCR Evaluation for Tamil Text Extraction
This project focuses on evaluating the performance of multiple Optical Character Recognition (OCR) tools for extracting Tamil text from scanned PDF documents. The pipeline handles end-to-end processing — from PDF upload to image conversion, OCR application, and output storage.

Objective:
To compare and benchmark various OCR engines for Tamil script recognition in real-world scanned PDFs, considering:
Accuracy
Layout retention
Handling of complex scripts

Pipeline Overview:
Upload PDF
The user uploads a scanned Tamil-language PDF.

1)Convert PDF to Images
Each page is converted into an image using pdf2image.

2)Apply OCR
Multiple OCR engines are applied individually to each image:
Tesseract OCR (with Tamil language pack)
PaddleOCR
EasyOCR

3)Extract & Save Text

4)The recognized Tamil text is stored in .txt or .json format.

Each OCR's result is saved separately for comparison.

Tools & Libraries Used:
Component	Tool/Library:
PDF to Image	pdf2image
OCR Engine 1	Tesseract-OCR
OCR Engine 2	PaddleOCR
OCR Engine 3	EasyOCR
Image Processing	OpenCV, PIL
Output Formats	.txt, .json

Evaluation Metrics:
To evaluate each OCR engine:
Character Accuracy
Word Accuracy
Error Types (e.g., missed characters, incorrect glyphs)
Support for complex Tamil words and ligatures

Output Structure:
Copy
Edit
outputs/
│
├── tesseract_output/
│   ├── page_1.txt
│   ├── ...
│
├── easyocr_output/
│   ├── page_1.txt
│   ├── ...
│
├── paddleocr_output/
│   ├── page_1.txt
│   ├── ...
│
└── combined_results.json   # (optional merged format)
Setup Instructions
bash
Copy
Edit
# Install dependencies
pip install pdf2image pytesseract easyocr paddleocr

# (Linux)
apt install tesseract-ocr tesseract-ocr-tam
Future Work
Add layout-aware extraction using LayoutParser or MarkItDown

Visual comparison of OCR outputs

Integrate word-level confidence scoring



the next part of this project is to use differnt open source tool for detecting the layouts, tables, images etc present in the pds and images.
 
