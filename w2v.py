import os
import re
import jieba
from collections import Counter
from gensim.models import Word2Vec

def load_stopwords(file_path):
    # 加载停用词
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().split())
    return stopwords

def clean_text(text, stopwords):
    # 清洗文本并去除停用词
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 分词
    words = jieba.cut(text)
    # 去除停用词
    cleaned_text = [word for word in words if word not in stopwords]
    return cleaned_text

def process_files(directory, stopwords_file):
    # 处理目录下所有txt文件
    stopwords = load_stopwords(stopwords_file)
    processed_texts = []
    all_words = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='gb18030') as file:
                text = file.read()
                cleaned_text = clean_text(text, stopwords)
                processed_texts.append(cleaned_text)
                all_words.extend(cleaned_text)
    return processed_texts, all_words

def save_words(words, file_path):
    # 将词语按词频保存到txt文件
    word_freq = Counter(words)
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        for word, freq in sorted_words:
            file.write(f"{word} {freq}\n")

# 使用函数处理指定目录下的所有txt文件
directory = 'jyxstxtqj_downcc.com'  # 小说文件所在的目录路径
stopwords_file = 'cn_stopwords.txt'  # 停用词文件路径
processed_texts, all_words = process_files(directory, stopwords_file)

# 保存所有分词结果，按词频排序
words_file_path = 'all_words_sorted_by_frequency.txt'
save_words(all_words, words_file_path)

# 训练Word2Vec模型
model = Word2Vec(processed_texts, vector_size=300, window=8, min_count=5, workers=24)

# 查看一个词的向量
# example_word = '杨过'
# if example_word in model.wv:
#     print(f"Vector for '{example_word}': {model.wv[example_word]}")
# else:
#     print(f"'{example_word}' not in vocabulary.")

# 保存模型
model.save("word2vec_model.model")

