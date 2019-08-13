import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. 加载数据集
iris = load_iris()
# 2. 创建DataFrame
iris_df = pd.DataFrame(iris.data, columns=
    ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])

# 3. 设置目标值
iris_df['target'] = iris.target
# 4. 定义绘制的方法
def plot_iris(data, col1, col2):
    sb.lmplot(x=col1, y=col2, data=data, hue='target', fit_reg=False)
    plt.title('鸢尾花散点图')
    plt.show()

# 5. 调用方法绘图
plot_iris(iris_df, 'Petal_Length', 'Petal_Width')



