# 📝 OCR for Tamil LLM Training

This project evaluates multiple **Optical Character Recognition (OCR)** engines for **Tamil text extraction** from scanned PDFs. The goal is to build a reliable pipeline to support **training Large Language Models (LLMs)** in Tamil by ensuring high-quality text extraction from real-world documents.

---

## 🎯 Objective

To **compare and benchmark** different OCR engines for **Tamil script recognition**, with a focus on:

- 📈 Accuracy  
- 🧱 Layout retention  
- 🧩 Handling complex Tamil scripts and ligatures  

---

## 🔄 Pipeline Overview

### 1. 📥 Upload PDF  
Users provide scanned Tamil-language PDF documents.

### 2. 🖼️ Convert PDF to Images  
Each page is converted into high-resolution images using `pdf2image`.

### 3. 🔍 Apply OCR Engines  
Multiple OCR tools are run on each image:
- **Tesseract OCR** (with Tamil language pack)
- **PaddleOCR**
- **EasyOCR**

### 4. 💾 Extract & Store Text  
Recognized text is saved in `.txt` and optionally `.json` format for further analysis.  
Each OCR engine’s results are saved **separately for comparison**.

---

## 🧰 Tools & Libraries Used

| Component           | Tool / Library       |
|---------------------|----------------------|
| PDF to Image        | `pdf2image`          |
| OCR Engine 1        | `Tesseract-OCR`      |
| OCR Engine 2        | `PaddleOCR`          |
| OCR Engine 3        | `EasyOCR`            |
| Image Processing    | `OpenCV`, `PIL`      |
| Output Formats      | `.txt`, `.json`      |

---

## 📊 Evaluation Metrics

To compare OCR engines, the following metrics are used:

- ✅ **Character-Level Accuracy**
- ✅ **Word-Level Accuracy**
- ❌ **Error Analysis**: missed characters, incorrect glyphs
- 🔤 **Support for complex Tamil ligatures and words**

---

## 📁 Output Structure

outputs/
│
├── tesseract_output/
│ ├── page_1.txt
│ ├── ...
│
├── easyocr_output/
│ ├── page_1.txt
│ ├── ...
│
├── paddleocr_output/
│ ├── page_1.txt
│ ├── ...
│
└── combined_results.json # (optional merged comparison)


---

## ⚙️ Setup Instructions

```bash
# Install Python dependencies
pip install pdf2image pytesseract easyocr paddleocr

# Install Tesseract OCR with Tamil support (Linux)
sudo apt install tesseract-ocr tesseract-ocr-tam

pip install pdf2image pytesseract easyocr paddleocr
sudo apt install tesseract-ocr tesseract-ocr-tam

##Future Work
Integration of layout-aware OCR using tools like LayoutParser or MarkItDown

Visual comparison of OCR outputs across engines

Incorporation of word-level confidence scores

Creation of a benchmarking dashboard for evaluation metrics

##Next Big Step
The upcoming phase of this project involves automatic detection of structural elements in scanned documents such as:

Tables

Images

Multi-column layouts

Section headers and footers

These elements will be detected using advanced layout parsing tools such as:

LayoutParser

Donut (Document Understanding Transformer)

DocTR

PaddleLayout

This will enable structured and intelligent document parsing, improving dataset quality for downstream Tamil LLM training.

##Author
Risab Jain
Researcher and developer passionate about OCR, document AI, and multilingual LLM pipelines.
