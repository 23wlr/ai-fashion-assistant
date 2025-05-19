import os
import requests
from duckduckgo_search import DDGS

# 儲存資料夾
SAVE_FOLDER = r"C:\Users\USER\OneDrive\桌面\人工智慧\clothes_db"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 搜尋設定
SEARCH_TERM = "fashion outfit clothing"
NUM_IMAGES = 50

def download_images():
    with DDGS() as ddgs:
        results = ddgs.images(SEARCH_TERM, max_results=NUM_IMAGES)
        for i, result in enumerate(results):
            url = result["image"]
            try:
                img_data = requests.get(url, timeout=10).content
                file_path = os.path.join(SAVE_FOLDER, f"{i+1}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                print(f"✅ 下載完成: {file_path}")
            except Exception as e:
                print(f"❌ 下載失敗 ({url}): {e}")
    
    print(f"\n🎉 成功下載圖片到 {SAVE_FOLDER}")

if __name__ == "__main__":
    download_images()



