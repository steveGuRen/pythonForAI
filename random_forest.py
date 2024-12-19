from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
import time

# 记录起始时间
start_time = time.time()

# 加载数据集
data = fetch_california_housing()
X, y = data.data, data.target

# 将数据集转换为 DataFrame 以便查看
df = pd.DataFrame(X, columns=data.feature_names)
df['Target'] = y

# 打印前十条记录
print("前十条数据记录：")
print(df.head(10))

# 打印特征的数量
print(f"\n特征数量: {X.shape[1]}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印训练集和测试集条目数
print(f"训练集条目数: {X_train.shape[0]}")
print(f"测试集条目数: {X_test.shape[0]}")

# 初始化随机森林回归器
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 打印每个特征的重要性
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\n各特征的重要性：")
print(importance_df)

# 保存模型到指定文件夹
model_filename = './result/random_forest_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"\nModel saved to {model_filename}")

# 加载已保存的模型
loaded_rf_model = joblib.load(model_filename)

# 使用加载的模型进行预测
y_pred_loaded = loaded_rf_model.predict(X_test)

# 输出性能
print("\n性能指标：")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_loaded))
print("R^2 Score:", r2_score(y_test, y_pred_loaded))

# 记录结束时间并计算总时长
end_time = time.time()
total_time = end_time - start_time
print(f"\n程序运行总时长: {total_time:.2f} 秒")
