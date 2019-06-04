import os
import pytesseract
from PIL import Image

os.environ["TESSDATA_PREFIX"] = 'C://Program Files (x86)//Tesseract-OCR//tessdata'
print (os.environ["TESSDATA_PREFIX"])

pytesseract.pytesseract.tesseract_cmd = 'C://Program Files (x86)/Tesseract-OCR/tesseract.exe'
text = pytesseract.image_to_string(Image.open('d://temp/1.jpg'), lang='chi_sim')

print(text)
