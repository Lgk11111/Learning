"""
电影推荐系统 - 完整代码
文件名：movie_recommendation.py
"""
# 添加这两行在文件最开头
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import zipfile
import urllib.request
import os


class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.user_movie_matrix = None

    def download_data(self):
        """下载MovieLens数据集"""
        print("正在下载数据集...")
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        urllib.request.urlretrieve(url, "movielens.zip")

        with zipfile.ZipFile("movielens.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

        print("数据集下载完成！")

    def load_data(self):
        """加载数据"""
        self.movies = pd.read_csv("data/ml-latest-small/movies.csv")
        self.ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

        print("数据加载完成！")
        print(f"电影数量：{len(self.movies)}")
        print(f"评分数量：{len(self.ratings)}")
        print(f"用户数量：{self.ratings['userId'].nunique()}")

    def explore_data(self):
        """数据探索"""
        print("\n=== 数据探索 ===")

        # 评分统计
        print("\n1. 评分统计：")
        print(self.ratings['rating'].describe())

        # 可视化
        plt.figure(figsize=(15, 5))

        # 评分分布
        plt.subplot(1, 3, 1)
        self.ratings['rating'].hist(bins=10, edgecolor='black')
        plt.title('评分分布')
        plt.xlabel('评分')
        plt.ylabel('数量')

        # 用户评分数量分布
        plt.subplot(1, 3, 2)
        user_rating_counts = self.ratings['userId'].value_counts()
        user_rating_counts.hist(bins=30, edgecolor='black')
        plt.title('用户评分数量分布')
        plt.xlabel('评分数量')
        plt.ylabel('用户数')

        # 电影评分数量分布
        plt.subplot(1, 3, 3)
        movie_rating_counts = self.ratings['movieId'].value_counts()
        movie_rating_counts.hist(bins=30, edgecolor='black')
        plt.title('电影评分数量分布')
        plt.xlabel('评分数量')
        plt.ylabel('电影数')

        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300)
        plt.show()

    def preprocess_data(self):
        """数据预处理"""
        print("\n=== 数据预处理 ===")

        # 创建用户-电影矩阵
        self.user_movie_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        )

        print(f"用户-电影矩阵形状：{self.user_movie_matrix.shape}")

    def build_user_based_model(self):
        """基于用户的协同过滤"""
        print("\n=== 基于用户的协同过滤 ===")

        # 计算用户相似度
        user_similarity = cosine_similarity(self.user_movie_matrix.fillna(0))

        # 推荐函数
        def recommend_for_user(user_id, n=5):
            similar_users = user_similarity[user_id - 1]
            similar_indices = similar_users.argsort()[-10:-1][::-1]

            recommendations = set()
            for idx in similar_indices:
                user_ratings = self.user_movie_matrix.iloc[idx]
                top_movies = user_ratings[user_ratings > 4].index.tolist()
                recommendations.update(top_movies)

            # 排除已看过的
            user_watched = self.user_movie_matrix.iloc[user_id - 1]
            watched_movies = user_watched[user_watched.notna()].index.tolist()
            recommendations = [m for m in recommendations if m not in watched_movies]

            return list(recommendations)[:n]

        # 测试
        test_user = 1
        recommendations = recommend_for_user(test_user, 3)

        print(f"\n为用户{test_user}推荐的电影：")
        for movie_id in recommendations:
            movie_info = self.movies[self.movies['movieId'] == movie_id]
            if len(movie_info) > 0:
                print(f"  - {movie_info['title'].values[0]}")

        return user_similarity

    def build_item_based_model(self):
        """基于电影的协同过滤"""
        print("\n=== 基于电影的协同过滤 ===")

        # 计算电影相似度
        movie_similarity = cosine_similarity(self.user_movie_matrix.fillna(0).T)

        def recommend_similar_movies(movie_id, n=3):
            movie_idx = list(self.user_movie_matrix.columns).index(movie_id)
            similar_movies = movie_similarity[movie_idx]
            similar_indices = similar_movies.argsort()[-n - 1:-1][::-1]

            similar_movie_ids = [self.user_movie_matrix.columns[i] for i in similar_indices]

            return similar_movie_ids

        # 测试
        test_movie = 1  # Toy Story
        similar_movies = recommend_similar_movies(test_movie, 3)

        print(f"\n与《Toy Story》相似的电影：")
        for movie_id in similar_movies:
            movie_info = self.movies[self.movies['movieId'] == movie_id]
            if len(movie_info) > 0:
                print(f"  - {movie_info['title'].values[0]}")

        return movie_similarity

    def evaluate_models(self):
        """评估模型"""
        print("\n=== 模型评估 ===")

        # 划分训练集和测试集
        train_data, test_data = train_test_split(
            self.ratings,
            test_size=0.2,
            random_state=42
        )

        # 简单评估：计算基线准确率
        def calculate_baseline_accuracy(train_df, test_df):
            correct = 0
            total = 0

            for _, row in test_df.iterrows():
                user_id = row['userId']
                actual_rating = row['rating']

                # 获取用户平均评分作为预测
                user_ratings = train_df[train_df['userId'] == user_id]['rating']
                if len(user_ratings) > 0:
                    predicted_rating = user_ratings.mean()

                    # 简单分类：喜欢(>3.5) vs 不喜欢(≤3.5)
                    if (predicted_rating > 3.5 and actual_rating > 3.5) or \
                            (predicted_rating <= 3.5 and actual_rating <= 3.5):
                        correct += 1
                    total += 1

            return correct / total if total > 0 else 0

        accuracy = calculate_baseline_accuracy(train_data, test_data)
        print(f"基线模型准确率：{accuracy:.2%}")

        return accuracy

    def run_full_analysis(self):
        """运行完整分析"""
        print("开始电影推荐系统分析...")

        # 1. 下载数据
        if not os.path.exists("data/ml-latest-small"):
            self.download_data()

        # 2. 加载数据
        self.load_data()

        # 3. 数据探索
        self.explore_data()

        # 4. 数据预处理
        self.preprocess_data()

        # 5. 构建模型
        user_similarity = self.build_user_based_model()
        movie_similarity = self.build_item_based_model()

        # 6. 评估模型
        accuracy = self.evaluate_models()

        print("\n=== 分析完成 ===")
        print(f"最终准确率：{accuracy:.2%}")
        print("结果已保存到文件！")


# 运行程序
if __name__ == "__main__":
    recommender = MovieRecommender()
    recommender.run_full_analysis()