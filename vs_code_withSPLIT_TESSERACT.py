import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ✅ SETUP: Tesseract & Poppler paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\jainr\OneDrive\Desktop\1_internship\downloadssss\tesseract.exe"
poppler_path = r"C:\Users\jainr\OneDrive\Desktop\1_internship\downloadssss\poppler-24.08.0\Library\bin"

# ✅ Input folder with PDFs
folder_path = r"C:\Users\jainr\OneDrive\Desktop\1_internship\TAMIL OCR\tamil_tesseract_ocrr\Tamil LLM\Class 6"
output_base = os.path.join(folder_path, "outputs")  # Output directory

# ✅ Tamil OCR function
def extract_tamil_text(img):
    configs = [
        '--psm 6 -l tam',
        '--psm 3 -l tam',
        '--psm 6 -l tam+eng',
        '--psm 4 -l tam',
    ]
    best_text = ""
    for config in configs:
        try:
            text = pytesseract.image_to_string(img, config=config)
            if text.strip() and len(text.strip()) > len(best_text):
                best_text = text.strip()
        except Exception as e:
            print(f"⚠️ OCR failed with config '{config}': {e}")
    return best_text

# ✅ Middle gap detector
def is_split_layout_by_gap(img):
    width, height = img.size
    middle_strip = img.crop((width // 2 - 10, 0, width // 2 + 10, height)).convert("L")
    white_pixels = sum(1 for px in middle_strip.getdata() if px > 200)
    total_pixels = middle_strip.size[0] * middle_strip.size[1]
    return (white_pixels / total_pixels) > 0.9

# ✅ Gather PDFs
pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
print(f"📁 Found {len(pdf_files)} PDF(s) to process.\n")

# ✅ Process each PDF
for pdf_index, pdf_file in enumerate(pdf_files, start=1):
    pdf_path = os.path.join(folder_path, pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]
    print(f"\n📄 [{pdf_index}] Processing: {pdf_file}")

    try:
        # Convert PDF to images
        pages = convert_from_path(pdf_path, dpi=150, poppler_path=poppler_path)
        print(f"📄 Total pages: {len(pages)}")

        results = []

        for page_num, img in enumerate(pages, start=1):
            print(f"🔍 Page {page_num}/{len(pages)}")
            is_split = is_split_layout_by_gap(img)

            result = f"=== {pdf_file} | Page {page_num} ===\n\n"

            if is_split:
                print("📐 Layout: SPLIT")
                width, height = img.size
                left_half = img.crop((0, 0, width // 2, height))
                right_half = img.crop((width // 2, 0, width, height))

                left_text = extract_tamil_text(left_half)
                right_text = extract_tamil_text(right_half)

                result += "[Layout: Split Page]\n\n"
                result += f"[Left Half]\n{left_text}\n\n[Right Half]\n{right_text}\n\n"
            else:
                print("🧾 Layout: FULL")
                full_text = extract_tamil_text(img)
                result += "[Layout: Full Page]\n\n"
                result += f"{full_text}\n\n"

            results.append(result)

        # ✅ Create output folder for this PDF
        pdf_output_dir = os.path.join(output_base, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        # ✅ Save output to text file inside that folder
        output_path = os.path.join(pdf_output_dir, f"{pdf_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Tamil OCR Output for: {pdf_file}\n")
            f.write("=" * 60 + "\n\n")
            f.write("\n".join(results) if results else "⚠️ No text extracted.")

        print(f"✅ Saved OCR result to:\n📄 {output_path}")

    except Exception as e:
        print(f"❌ Error processing {pdf_file}: {e}")





# import os
# from pdf2image import convert_from_path
# from PIL import Image
# import pytesseract

# # ✅ SETUP: Tesseract & Poppler paths (update as needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\jainr\OneDrive\Desktop\1_internship\downloadssss\tesseract.exe"
# poppler_path = r"C:\Users\jainr\OneDrive\Desktop\1_internship\downloadssss\poppler-24.08.0\Library\bin"

# # ✅ Input folder with PDFs
# folder_path = r"C:\Users\jainr\OneDrive\Desktop\1_internship\TAMIL OCR\tamil_tesseract_ocrr\Tamil LLM\Class 6"

# # ✅ Output file (stored in same folder)
# output_file = os.path.join(folder_path, "all_pdfs_split_half_ocr_output.txt")

# # ✅ Tamil OCR with fallback configs
# def extract_tamil_text(img):
#     configs = [
#         '--psm 6 -l tam',
#         '--psm 3 -l tam',
#         '--psm 6 -l tam+eng',
#         '--psm 4 -l tam',
#     ]
#     best_text = ""
#     for config in configs:
#         try:
#             text = pytesseract.image_to_string(img, config=config)
#             if text.strip() and len(text.strip()) > len(best_text):
#                 best_text = text.strip()
#         except Exception as e:
#             print(f"⚠️ OCR failed with config '{config}': {e}")
#     return best_text

# # ✅ Get all PDF files
# pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
# print(f"📁 Found {len(pdf_files)} PDFs in: {folder_path}")

# results = []

# # ✅ Process each PDF
# for pdf_index, pdf_file in enumerate(pdf_files, start=1):
#     pdf_path = os.path.join(folder_path, pdf_file)
#     print(f"\n📄 Processing PDF {pdf_index}/{len(pdf_files)}: {pdf_file}")

#     try:
#         pages = convert_from_path(pdf_path, dpi=150, poppler_path=poppler_path)
#         print(f"📄 {len(pages)} pages found")

#         for page_num, img in enumerate(pages, start=1):
#             print(f"🔍 Page {page_num}/{len(pages)}")

#             width, height = img.size
#             left_half = img.crop((0, 0, width // 2, height))
#             right_half = img.crop((width // 2, 0, width, height))

#             # QUICK LAYOUT CHECK using fast OCR
#             left_sample = pytesseract.image_to_string(left_half, config="--psm 6 -l tam")
#             right_sample = pytesseract.image_to_string(right_half, config="--psm 6 -l tam")
#             full_sample = pytesseract.image_to_string(img, config="--psm 6 -l tam")

#             is_split = (
#                 len(left_sample.strip()) > 50 and
#                 len(right_sample.strip()) > 50 and
#                 abs(len(left_sample) - len(right_sample)) < 0.5 * max(len(left_sample), len(right_sample))
#             )

#             result = f"=== {pdf_file} | Page {page_num} ===\n\n"

#             if is_split:
#                 print("📐 Split layout detected")
#                 left_text = extract_tamil_text(left_half)
#                 right_text = extract_tamil_text(right_half)
#                 result += "[Layout: Split Page]\n\n"
#                 result += f"[Left Half]\n{left_text}\n\n[Right Half]\n{right_text}\n\n"
#             else:
#                 print("🧾 Full page layout detected")
#                 full_text = extract_tamil_text(img)
#                 result += "[Layout: Full Page]\n\n"
#                 result += f"{full_text}\n\n"

#             results.append(result)

#     except Exception as e:
#         print(f"❌ Error processing {pdf_file}: {e}")

# # ✅ Save results to file
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write("Tamil Textbook OCR Output (Auto Layout Detection)\n")
#     f.write("=" * 70 + "\n\n")
#     f.write("\n".join(results) if results else "⚠️ No text extracted.")

# print(f"\n✅ OCR complete! Results saved to:\n📁 {output_file}")