import pytesseract
from PIL import Image
import requests
from io import BytesIO

# 通过URL加载图片
image_url = "https://example.com/some_image.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# 使用pytesseract进行OCR
text_from_image = pytesseract.image_to_string(img)

# 将提取到的文字添加到帖子数据中
post_image_text = clean_text(text_from_image)  # 同样使用之前的clean_text函数清洗
post.append(post_image_text)


cursor.execute('INSERT INTO posts (title, body, image_text, created_at) VALUES (%s, %s, %s, %s)',
               (post_title, post_body, post_image_text, created_at))
