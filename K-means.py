import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.spatial.distance import cosine

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def load_model(model_path):
    # 加载预训练的Word2Vec模型
    model = Word2Vec.load(model_path)
    return model

def get_word_vectors(model, words):
    # 从模型中获取词向量
    word_vectors = np.array([model.wv[word] for word in words if word in model.wv])
    present_words = [word for word in words if word in model.wv]
    return word_vectors, present_words

def perform_clustering(word_vectors, num_clusters):
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(word_vectors)
    labels = kmeans.labels_
    return labels

def cosine_similarity(vec1, vec2):
    # 计算余弦相似度
    return 1 - cosine(vec1, vec2)

def visualize_clusters(words, word_vectors, labels):
    # 使用t-SNE进行降维并可视化聚类结果
    tsne = TSNE(n_components=2, random_state=0, perplexity=min(5, len(words) - 1))
    reduced_vectors = tsne.fit_transform(word_vectors)

    # 设置中文字体
    font_path = "C:/Windows/Fonts/simhei.ttf" 
    font = FontProperties(fname=font_path)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='viridis')
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontproperties=font)
    plt.title("Word Clusters Visualization", fontproperties=font)
    plt.xlabel("TSNE Component 1", fontproperties=font)
    plt.ylabel("TSNE Component 2", fontproperties=font)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

def main():
    model_path = 'word2vec_model.model'
    words = [
        '杨过', '小龙女', '韦小宝', '张无忌', '赵敏', '郭靖', '黄蓉', '洪七公', '欧阳锋', 
        '黄药师', '周伯通', '段誉', '乔峰', '虚竹', '令狐冲', '任我行', '东方不败', 
        '九阳神功', '降龙十八掌', '独孤九剑', '乾坤大挪移', '打狗棒法', '一阳指'
    ]
    num_clusters = 8  # 调整聚类数量

    model = load_model(model_path)
    word_vectors, present_words = get_word_vectors(model, words)
    labels = perform_clustering(word_vectors, num_clusters)
    
    for word, label in zip(present_words, labels):
        print(f"{word}: Cluster {label}")
    
    # 计算余弦相似度
    for i, word1 in enumerate(present_words):
        for j, word2 in enumerate(present_words):
            if i < j:
                sim = cosine_similarity(model.wv[word1], model.wv[word2])
                print(f"Cosine similarity between '{word1}' and '{word2}': {sim:.4f}")
    
    # 可视化聚类结果
    visualize_clusters(present_words, word_vectors, labels)

if __name__ == "__main__":
    main()
