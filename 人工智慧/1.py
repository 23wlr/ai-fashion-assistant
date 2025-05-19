import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import pickle

# 1. 定義資料夾與載入模型
DB_IMAGE_FOLDER = 'clothes_db/'  # 你的衣服資料庫資料夾，裡面放約50張圖片
MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # 去掉分類層，做特徵向量提取

# 2. 特徵提取函式
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = MODEL.predict(x)
    return features.flatten()

# 3. 建立資料庫特徵向量（只用一次，之後可以存pickle）
def build_feature_database():
    features_list = []
    filenames = []
    for fname in os.listdir(DB_IMAGE_FOLDER):
        if fname.lower().endswith(('.jpg','.png')):
            fpath = os.path.join(DB_IMAGE_FOLDER, fname)
            feat = extract_features(fpath)
            features_list.append(feat)
            filenames.append(fname)
    features_np = np.array(features_list)
    # 存起來以便下次快速使用
    with open('features.pkl', 'wb') as f:
        pickle.dump((features_np, filenames), f)
    return features_np, filenames

# 4. 載入特徵資料庫
def load_feature_database():
    if os.path.exists('features.pkl'):
        with open('features.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return build_feature_database()

# 5. 推薦函式，用KNN找相似向量
def recommend(upload_feat, features_np, filenames, top_k=5):
    knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
    knn.fit(features_np)
    distances, indices = knn.kneighbors([upload_feat])
    results = [filenames[i] for i in indices[0]]
    return results

# 6. Streamlit介面
def main():
    st.title("AI時尚穿搭助手")
    st.write("上傳你的穿搭照片，推薦相似衣服搭配")

    uploaded_file = st.file_uploader("請上傳服裝照片", type=['jpg','png'])
    if uploaded_file is not None:
        # 讀入圖片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(img, channels="BGR")

        # 存檔做特徵提取
        temp_file = 'temp.jpg'
        cv2.imwrite(temp_file, img)

        # 取得特徵向量
        upload_feat = extract_features(temp_file)

        # 載入資料庫特徵
        features_np, filenames = load_feature_database()

        # 推薦
        recommended = recommend(upload_feat, features_np, filenames)

        st.write("推薦的相似服裝：")
        for fname in recommended:
            fpath = os.path.join(DB_IMAGE_FOLDER, fname)
            st.image(fpath, width=150)

if __name__ == '__main__':
    main()
