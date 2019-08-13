from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 字典特征提取
#
# datas = [{'city': '北京','temperature':100},
# {'city': '上海','temperature':60},
# {'city': '深圳','temperature':30}]
#
# # 1. 创建字典特征提取器
# transfer = DictVectorizer(sparse=False)
#
# # 2. 进行字典特征提取
# new_datas = transfer.fit_transform(datas)
# print(new_datas)
# # 获取每一个特征的名称
# print(transfer.get_feature_names())
#

# 文本特征提取: 对文本进行特征值化
# texts = ["life is short,i like like python",
# "life is too long,i dislike python"]
# # texts = ["人生 苦短，我 喜欢 Python","生活 太 长久，我 不喜欢 Python"]
#
# # 创建文本特征提取器对象
# transfer = CountVectorizer()
# # 进行文本特征提取
# new_data = transfer.fit_transform(texts)
# # print(new_data) # 稀疏矩阵
# print(new_data.toarray())
# # 获取单词列表
# print(transfer.get_feature_names())


# jieba分词

# words = jieba.cut('我爱北京天安门')
# # print(words)
# # for word in words:
# #     print(word)
# print(list(words))

def cut_word(text):
    """返回分词后文本内容"""
    return ' '.join(list(jieba.cut(text)))

# text = cut_word('我爱北京天安门')
# print(text)


def chinese_text_extract():
    """中文特征提取"""
    # 1. 准备中文数据
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    # 2. 进行中文分词
    texts = [cut_word(text) for text in data]
    # print(texts)
    # 3. 创建文本特征提取器
    transfer = CountVectorizer(stop_words=['了解', '一种', '不会'])

    # 4. 进行文本特征提取
    new_texts = transfer.fit_transform(texts)
    # 5. 打印结果
    print(new_texts.toarray())
    print(transfer.get_feature_names())

def chinese_text_tiidf():
    """中文特征提取"""
    # 1. 准备中文数据
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    # 2. 进行中文分词
    texts = [cut_word(text) for text in data]
    # print(texts)
    # 3. 创建文本特征提取器
    transfer = TfidfVectorizer(stop_words=['了解', '一种', '不会'])

    # 4. 进行文本特征提取
    new_texts = transfer.fit_transform(texts)
    # 5. 打印结果
    print(new_texts.toarray())
    print(transfer.get_feature_names())


if __name__ == '__main__':
    # chinese_text_extract()
    chinese_text_tiidf()