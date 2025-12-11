import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import time
import warnings
import os

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

class ClusteringAnalyzer:
    def __init__(self):
        self.results = {}
        
    def check_data_file(self, filename):
        """檢查資料檔案格式"""
        if not os.path.exists(filename):
            print(f"檔案 {filename} 不存在")
            return None
            
        try:
            data = pd.read_csv(filename)
            print(f"\n檔案 {filename} 內容:")
            print(f"形狀: {data.shape}")
            print(f"欄位: {data.columns.tolist()}")
            print("\n前5行:")
            print(data.head())
            return data
        except Exception as e:
            print(f"讀取 {filename} 時出錯: {e}")
            return None
    
    def smart_load_data(self, filename, expected_clusters):
        """
        智能載入資料，自動處理不同格式
        """
        print(f"\n嘗試載入 {filename}...")
        
        if not os.path.exists(filename):
            print(f"檔案 {filename} 不存在，使用模擬資料")
            return None, None
            
        try:
            data = pd.read_csv(filename)
            print(f"成功讀取，資料形狀: {data.shape}")
            print(f"欄位名稱: {data.columns.tolist()}")
            
            # 判斷資料格式
            n_columns = data.shape[1]
            
            # 情況1: 至少有3欄，假設最後一欄是標籤
            if n_columns >= 3:
                X_columns = data.columns[:-1]
                y_column = data.columns[-1]
                X = data[X_columns].values
                y = data[y_column].values
                
                print(f"使用前 {n_columns-1} 欄作為特徵: {list(X_columns)}")
                print(f"使用最後1欄作為標籤: {y_column}")
                
                unique_labels = np.unique(y)
                print(f"標籤種類: {unique_labels} (共{len(unique_labels)}種)")
                
                # 確保標籤從0開始
                if min(unique_labels) != 0:
                    y = y - min(unique_labels)
                    print("已調整標籤使其從0開始")
                
                return X, y
            
            # 情況2: 只有2欄，使用聚類產生標籤
            elif n_columns == 2:
                print("只有2欄，使用K-means產生模擬標籤")
                X = data.values
                
                # 根據expected_clusters產生標籤
                kmeans = KMeans(n_clusters=expected_clusters, random_state=42)
                y = kmeans.fit_predict(X)
                
                return X, y
            
            # 情況3: 其他情況
            else:
                print(f"無法處理的資料格式，欄位數: {n_columns}")
                return None, None
                
        except Exception as e:
            print(f"載入 {filename} 時出錯: {e}")
            return None, None
    
    def load_banana_data(self, filepath='banana.csv'):
        """
        載入banana資料集
        """
        # 先嘗試智能載入
        X, y = self.smart_load_data(filepath, expected_clusters=2)
        
        if X is not None and y is not None:
            print(f"成功從 {filepath} 載入 {len(X)} 筆資料")
            return X, y
        
        # 如果載入失敗，使用模擬資料
        print(f"使用模擬banana資料")
        np.random.seed(42)
        n_samples = 300
        
        # 模擬半月形資料
        theta = np.linspace(np.pi/2, 3*np.pi/2, n_samples//2)
        r = 0.5 + 0.2 * np.random.randn(n_samples//2)
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta) + 0.5
        
        theta = np.linspace(-np.pi/2, np.pi/2, n_samples//2)
        r = 0.5 + 0.2 * np.random.randn(n_samples//2)
        x2 = r * np.cos(theta) + 2.5
        y2 = r * np.sin(theta) - 0.5
        
        X = np.vstack([np.column_stack([x1, y1]), 
                      np.column_stack([x2, y2])])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        return X, y
    
    def load_sizes3_data(self, filepath='sizes3.csv'):
        """
        載入sizes3資料集
        """
        # 先嘗試智能載入
        X, y = self.smart_load_data(filepath, expected_clusters=4)
        
        if X is not None and y is not None:
            print(f"成功從 {filepath} 載入 {len(X)} 筆資料")
            return X, y
        
        # 如果載入失敗，使用模擬資料
        print(f"使用模擬sizes3資料")
        np.random.seed(42)
        n_samples = 400
        
        # 創建4個不同大小的群集
        centers = [(1, 1), (1, 5), (5, 1), (5, 5)]
        scales = [0.2, 0.4, 0.6, 0.8]
        
        X_list = []
        y_list = []
        
        for i, (center, scale) in enumerate(zip(centers, scales)):
            x = np.random.normal(center[0], scale, n_samples//4)
            y = np.random.normal(center[1], scale, n_samples//4)
            X_list.append(np.column_stack([x, y]))
            y_list.append(np.full(n_samples//4, i))
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        return X, y
    
    def cluster_accuracy(self, y_true, y_pred):
        """計算分群準確率"""
        if len(np.unique(y_pred)) <= 1:
            return 0.0
            
        conf_matrix = confusion_matrix(y_true, y_pred)
        row_indices = np.arange(len(conf_matrix))
        col_indices = np.argmax(conf_matrix, axis=1)
        aligned_accuracy = conf_matrix[row_indices, col_indices].sum() / len(y_true)
        return aligned_accuracy
    
    def cluster_entropy(self, y_true, y_pred):
        """計算分群熵值"""
        if len(np.unique(y_pred)) <= 1:
            return 1.0
            
        return 1.0 - normalized_mutual_info_score(y_true, y_pred)
    
    def calculate_sse(self, X, labels, centers=None):
        """計算SSE"""
        unique_labels = np.unique(labels[labels != -1])
        sse = 0.0
        
        if centers is None:
            centers = []
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 0:
                    centers.append(np.mean(X[mask], axis=0))
                else:
                    centers.append(np.zeros(X.shape[1]))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if np.sum(mask) > 0:
                sse += np.sum(np.linalg.norm(X[mask] - centers[i], axis=1) ** 2)
        
        return sse
    
    def calculate_metrics(self, X, labels_true, labels_pred, algorithm_name, elapsed_time, centers=None):
        """計算評估指標"""
        metrics = {
            'algorithm': algorithm_name,
            'time': elapsed_time,
        }
        
        metrics['SSE'] = self.calculate_sse(X, labels_pred, centers)
        metrics['Accuracy'] = self.cluster_accuracy(labels_true, labels_pred)
        metrics['Entropy'] = self.cluster_entropy(labels_true, labels_pred)
        
        return metrics
    
    def plot_clusters(self, X, y_true, y_pred, title, markers, ax):
        """繪製分群結果"""
        unique_labels = np.unique(y_pred)
        
        if -1 in unique_labels:
            mask = y_pred == -1
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c='gray', marker='x', s=20, alpha=0.6, label='Noise')
            unique_labels = unique_labels[unique_labels != -1]
        
        for i, label in enumerate(unique_labels):
            mask = y_pred == label
            if i < len(markers):
                marker = markers[i]
            else:
                marker = markers[i % len(markers)]
            
            ax.scatter(X[mask, 0], X[mask, 1], 
                      marker=marker, s=50, alpha=0.7, 
                      label=f'Cluster {label}')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(unique_labels) <= 6:
            ax.legend(loc='upper right')
    
    def analyze_banana(self):
        """分析banana資料集"""
        print("="*60)
        print("BANANA資料集分析 (分成2群)")
        print("="*60)
        
        # 檢查檔案
        data_info = self.check_data_file('banana.csv')
        
        # 載入資料
        X, y_true = self.load_banana_data()
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 繪圖
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 原始資料
        axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=50, alpha=0.7)
        axes[0].set_title(f'原始資料 (2群, {len(X)}筆)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        markers = ['+', 'o']
        
        # K-means
        start_time = time.time()
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        y_pred_kmeans = kmeans.fit_predict(X_scaled)
        time_kmeans = time.time() - start_time
        centers_kmeans = kmeans.cluster_centers_
        
        self.plot_clusters(X, y_true, y_pred_kmeans, 'K-means分群結果', markers, axes[1])
        
        # 階層式分群
        start_time = time.time()
        hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
        y_pred_hier = hierarchical.fit_predict(X_scaled)
        time_hier = time.time() - start_time
        
        self.plot_clusters(X, y_true, y_pred_hier, '階層式分群結果', markers, axes[2])
        
        # DBSCAN
        dbscan_params = [(0.3, 5), (0.4, 5), (0.5, 5)]
        dbscan_titles = ['DBSCAN (eps=0.3)', 'DBSCAN (eps=0.4)', 'DBSCAN (eps=0.5)']
        
        for idx, ((eps, min_samples), title) in enumerate(zip(dbscan_params, dbscan_titles), 3):
            start_time = time.time()
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred_dbscan = dbscan.fit_predict(X_scaled)
            time_dbscan = time.time() - start_time
            
            self.plot_clusters(X, y_true, y_pred_dbscan, title, markers + ['x'], axes[idx])
        
        plt.suptitle('Banana資料集 - 分群演算法比較', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 計算指標
        print("\n分群結果比較:")
        print("-"*70)
        print(f"{'演算法':<20} {'時間(s)':<10} {'SSE':<12} {'Accuracy':<10} {'Entropy':<10}")
        print("-"*70)
        
        # K-means
        metrics_kmeans = self.calculate_metrics(X_scaled, y_true, y_pred_kmeans, 
                                               'K-means', time_kmeans, centers_kmeans)
        print(f"{'K-means':<20} {metrics_kmeans['time']:<10.4f} "
              f"{metrics_kmeans['SSE']:<12.4f} "
              f"{metrics_kmeans['Accuracy']:<10.4f} "
              f"{metrics_kmeans['Entropy']:<10.4f}")
        
        # 階層式分群
        metrics_hier = self.calculate_metrics(X_scaled, y_true, y_pred_hier, 
                                             '階層式分群', time_hier)
        print(f"{'階層式分群':<20} {metrics_hier['time']:<10.4f} "
              f"{metrics_hier['SSE']:<12.4f} "
              f"{metrics_hier['Accuracy']:<10.4f} "
              f"{metrics_hier['Entropy']:<10.4f}")
        
        # DBSCAN
        print(f"\nDBSCAN參數比較:")
        print("-"*70)
        print(f"{'參數':<20} {'時間(s)':<10} {'SSE':<12} {'Accuracy':<10} {'Entropy':<10} {'群集數':<8}")
        print("-"*70)
        
        for (eps, min_samples), title in zip(dbscan_params, dbscan_titles):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred_dbscan = dbscan.fit_predict(X_scaled)
            start_time = time.time()
            y_pred_dbscan = dbscan.fit_predict(X_scaled)
            time_dbscan = time.time() - start_time
            
            metrics_dbscan = self.calculate_metrics(X_scaled, y_true, y_pred_dbscan,
                                                   f'DBSCAN(eps={eps})', time_dbscan)
            n_clusters = len(np.unique(y_pred_dbscan[y_pred_dbscan != -1]))
            
            print(f"{title:<20} {time_dbscan:<10.4f} "
                  f"{metrics_dbscan['SSE']:<12.4f} "
                  f"{metrics_dbscan['Accuracy']:<10.4f} "
                  f"{metrics_dbscan['Entropy']:<10.4f} "
                  f"{n_clusters:<8}")
        
        return X_scaled, y_true
    
    def analyze_sizes3(self):
        """分析sizes3資料集"""
        print("\n" + "="*60)
        print("SIZES3資料集分析 (分成4群)")
        print("="*60)
        
        # 檢查檔案
        data_info = self.check_data_file('sizes3.csv')
        
        # 載入資料
        X, y_true = self.load_sizes3_data()
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 繪圖
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 原始資料
        axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=50, alpha=0.7)
        axes[0].set_title(f'原始資料 (4群, {len(X)}筆)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        markers = ['1', '2', '3', '4']
        
        # K-means
        start_time = time.time()
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        y_pred_kmeans = kmeans.fit_predict(X_scaled)
        time_kmeans = time.time() - start_time
        centers_kmeans = kmeans.cluster_centers_
        
        self.plot_clusters(X, y_true, y_pred_kmeans, 'K-means分群結果', markers, axes[1])
        
        # 階層式分群
        start_time = time.time()
        hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
        y_pred_hier = hierarchical.fit_predict(X_scaled)
        time_hier = time.time() - start_time
        
        self.plot_clusters(X, y_true, y_pred_hier, '階層式分群結果', markers, axes[2])
        
        # DBSCAN
        dbscan_params = [(0.2, 10), (0.3, 10), (0.4, 10)]
        dbscan_titles = ['DBSCAN (eps=0.2)', 'DBSCAN (eps=0.3)', 'DBSCAN (eps=0.4)']
        
        for idx, ((eps, min_samples), title) in enumerate(zip(dbscan_params, dbscan_titles), 3):
            start_time = time.time()
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred_dbscan = dbscan.fit_predict(X_scaled)
            time_dbscan = time.time() - start_time
            
            self.plot_clusters(X, y_true, y_pred_dbscan, title, markers + ['x'], axes[idx])
        
        plt.suptitle('Sizes3資料集 - 分群演算法比較', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 計算指標
        print("\n分群結果比較:")
        print("-"*70)
        print(f"{'演算法':<20} {'時間(s)':<10} {'SSE':<12} {'Accuracy':<10} {'Entropy':<10}")
        print("-"*70)
        
        # K-means
        metrics_kmeans = self.calculate_metrics(X_scaled, y_true, y_pred_kmeans, 
                                               'K-means', time_kmeans, centers_kmeans)
        print(f"{'K-means':<20} {metrics_kmeans['time']:<10.4f} "
              f"{metrics_kmeans['SSE']:<12.4f} "
              f"{metrics_kmeans['Accuracy']:<10.4f} "
              f"{metrics_kmeans['Entropy']:<10.4f}")
        
        # 階層式分群
        metrics_hier = self.calculate_metrics(X_scaled, y_true, y_pred_hier, 
                                             '階層式分群', time_hier)
        print(f"{'階層式分群':<20} {metrics_hier['time']:<10.4f} "
              f"{metrics_hier['SSE']:<12.4f} "
              f"{metrics_hier['Accuracy']:<10.4f} "
              f"{metrics_hier['Entropy']:<10.4f}")
        
        # DBSCAN
        print(f"\nDBSCAN參數比較:")
        print("-"*70)
        print(f"{'參數':<20} {'時間(s)':<10} {'SSE':<12} {'Accuracy':<10} {'Entropy':<10} {'群集數':<8}")
        print("-"*70)
        
        for (eps, min_samples), title in zip(dbscan_params, dbscan_titles):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            start_time = time.time()
            y_pred_dbscan = dbscan.fit_predict(X_scaled)
            time_dbscan = time.time() - start_time
            
            metrics_dbscan = self.calculate_metrics(X_scaled, y_true, y_pred_dbscan,
                                                   f'DBSCAN(eps={eps})', time_dbscan)
            n_clusters = len(np.unique(y_pred_dbscan[y_pred_dbscan != -1]))
            
            print(f"{title:<20} {time_dbscan:<10.4f} "
                  f"{metrics_dbscan['SSE']:<12.4f} "
                  f"{metrics_dbscan['Accuracy']:<10.4f} "
                  f"{metrics_dbscan['Entropy']:<10.4f} "
                  f"{n_clusters:<8}")
        
        return X_scaled, y_true

def main():
    """主程式"""
    print("群聚分析作業 - 資料探勘")
    print("="*60)
    
    analyzer = ClusteringAnalyzer()
    
    # 1. 先檢查資料檔案
    print("\n[資料檔案檢查]")
    print("-"*40)
    
    banana_data = analyzer.check_data_file('banana.csv')
    sizes3_data = analyzer.check_data_file('sizes3.csv')
    
    # 2. 分析banana資料集
    print("\n" + "="*60)
    print("[任務1] Banana資料集分析")
    print("="*60)
    
    X_banana_scaled, y_banana = analyzer.analyze_banana()
    
    # 3. 分析sizes3資料集
    print("\n" + "="*60)
    print("[任務2] Sizes3資料集分析")
    print("="*60)
    
    X_sizes3_scaled, y_sizes3 = analyzer.analyze_sizes3()
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()