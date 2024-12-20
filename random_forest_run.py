import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 加载数据集
data = fetch_california_housing()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 保存模型到指定文件夹
model_filename = './result/random_forest_model.pkl'
# 加载已保存的模型
loaded_rf_model = joblib.load(model_filename)

# 使用加载的模型进行预测
y_pred_loaded = loaded_rf_model.predict(X_test)

# 输出性能
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_loaded))
print("R^2 Score:", r2_score(y_test, y_pred_loaded))
