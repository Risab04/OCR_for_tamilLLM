# vs code 
import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from tkinter import Tk, filedialog

# ✅ Optional: Set tesseract & poppler path manually (especially on Windows)
# Example for Windows (update these paths as per your install location)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\jainr\OneDrive\Desktop\1_internship\downloadssss\tesseract.exe"
poppler_path = r"C:\Users\jainr\OneDrive\Desktop\1_internship\downloadssss\poppler-24.08.0\Library\bin"
 # Required only on Windows

# ✅ Step 1: Ask user to select a PDF file
print("📤 Please select a Tamil PDF file...")
root = Tk()
root.withdraw()  # Hide the GUI window
pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])

if not pdf_path:
    print("❌ No file selected. Exiting.")
    exit()

print(f"📄 Selected PDF: {pdf_path}")

# ✅ Step 2: Convert PDF to images
print("🔄 Converting PDF to images (first 20 pages)...")
pdf_images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=20, poppler_path=poppler_path if os.name == 'nt' else None)
print(f"✅ Converted {len(pdf_images)} pages")

# ✅ Step 3: Define OCR function
def extract_tamil_text(img):
    configs = [
        '--psm 6 -l tam',      
        '--psm 3 -l tam',      
        '--psm 6 -l tam+eng',  
        '--psm 8 -l tam',      
        '--psm 4 -l tam',      
    ]
    best_text = ""
    for config in configs:
        try:
            text = pytesseract.image_to_string(img, config=config)
            if text.strip() and len(text.strip()) > len(best_text):
                best_text = text.strip()
        except Exception as e:
            print(f"⚠️ Config '{config}' failed: {e}")
    return best_text

# ✅ Step 4: Process only first 10 pages
print("📝 Processing first 10 pages...")
extracted_text = []
pages_to_process = pdf_images[:10]

for i, img in enumerate(pages_to_process, start=1):
    print(f"\n🔍 OCR on Page {i}/{len(pdf_images)}")
    img = img.convert("RGB")
    img.save(f"page_{i}.png")
    try:
        text = extract_tamil_text(img)
        if text:
            extracted_text.append(f"=== Page {i} ===\n{text}\n")
            print(f"✅ Extracted {len(text)} characters")
        else:
            print("⚠️ No text found.")
    except Exception as e:
        print(f"❌ Error on page {i}: {e}")

# ✅ Step 5: Save results
output_filename = "tamil_pdf_tesseract_output.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("Tamil PDF OCR Results (Tesseract)\n")
    f.write("=" * 50 + "\n")
    f.write(f"PDF File: {os.path.basename(pdf_path)}\n")
    f.write(f"Pages Processed: {len(pdf_images)}\n\n")
    f.write("Extracted Text:\n")
    f.write("-" * 30 + "\n")
    f.write("\n".join(extracted_text) if extracted_text else "⚠️ No text extracted.")

print(f"\n📁 OCR complete! Output saved to: {output_filename}")

# ✅ Preview output
print("\n📋 Preview of extracted text:")
print("-" * 30)
preview = "\n".join(extracted_text)[:500]
print(preview + "..." if len(preview) == 500 else preview)

