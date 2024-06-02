import gensim
from scipy.spatial.distance import cosine

def cosine_similarity(vec1, vec2):
    # 计算余弦相似度
    return 1 - cosine(vec1, vec2)

def check_and_compute_similarity(model, word1, word2):
    # 检查词汇并计算余弦相似度
    if word1 in model.wv and word2 in model.wv:
        sim = cosine_similarity(model.wv[word1], model.wv[word2])
        print(f"Similarity between '{word1}' and '{word2}': {sim:.4f}")
    else:
        print(f"One or both words not in vocabulary: {word1}, {word2}")

def main(model_path):
    # 主函数：加载模型并验证几对词
    # 加载模型
    model = gensim.models.Word2Vec.load(model_path)
    
    # 定义要检查的词对
    pairs = [
        ("杨过", "小龙女"),
        ("杨过", "郭襄"),
        ("杨过", "林朝英"),
        ("杨过", "虚竹"),
        ("杨过", "王重阳"),
        ("杨过", "张无忌"),
    ]
    
    # 对每对词汇计算相似度
    for word1, word2 in pairs:
        check_and_compute_similarity(model, word1, word2)

if __name__ == "__main__":
    model_path = 'word2vec_model.model'  # 模型路径
    main(model_path)
