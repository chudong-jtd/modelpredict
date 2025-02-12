import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 加载数据集
data = pd.read_csv('a22.csv')

# 分离特征和目标变量
X = data.iloc[:, :-1]  # 所有行，除了最后一列
y = data.iloc[:, -1]   # 最后一列为目标变量

# 划分训练集和测试集（80% 训练集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 评估模型
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 保存模型和标准化器
joblib.dump(rf_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')