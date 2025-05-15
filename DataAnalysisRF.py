#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
sns.set(font="SimHei", style="white")

def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


def extract_numeric(df: pd.DataFrame) -> pd.DataFrame:

    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df

def train_and_evaluate(X: pd.DataFrame, Y: pd.Series, k: int = 5):
    """
    使用随机森林进行 k 折交叉验证训练，并计算指标、特征重要性。
    返回值：
        - eval_metrics: 字典，包含平均 R2、RMSE 和 MAE
        - feature_importances: DataFrame，包含每折的特征重要性
        - shap_means: DataFrame（随机森林可以支持 SHAP，若不需要可设为 NaN）
        - train_rmse_folds: 空列表（RF 不输出训练过程误差）
        - val_rmse_folds: list，每折的验证 RMSE
        - all_true: 所有验证集真实值
        - all_pred: 所有验证集预测值
    """
    # 清理缺失值
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    Y = Y.loc[X.index]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []
    feature_importances = pd.DataFrame()
    val_rmse_folds = []
    all_true, all_pred = [], []
    shap_means = pd.DataFrame()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        all_true.extend(y_val)
        all_pred.extend(y_pred)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        val_rmse_folds.append(rmse)

        # 特征重要性
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'fold': fold + 1
        })
        feature_importances = pd.concat([feature_importances, fold_importance], axis=0)

        # SHAP 计算（可选）：这里先留空
        fold_shap = pd.DataFrame({
            'feature': X.columns,
            'shap_mean': [np.nan] * X.shape[1],
            'fold': fold + 1
        })
        shap_means = pd.concat([shap_means, fold_shap], axis=0)

    eval_metrics = {
        'mean_r2': np.mean(r2_scores),
        'mean_rmse': np.mean(rmse_scores),
        'mean_mae': np.mean(mae_scores)
    }

    return eval_metrics, feature_importances, shap_means, [], val_rmse_folds, all_true, all_pred


def plot_true_vs_pred(all_true: list, all_pred: list, save_path: str = 'True_vs_Predicted_Values.png'):

    plt.figure(figsize=(10, 6))
    plt.scatter(all_true, all_pred, alpha=0.6)
    plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], '--r', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.savefig(save_path)
    plt.show()

def main():
    file_path = "模型数据.xlsx"
    sheet_name = "汇总"

    df = load_data(file_path, sheet_name)
    numeric_df = extract_numeric(df)

    feature_cols = ['渔业产业GDP（万元）', '捕捞产量', '渔业劳动力（人）', '池塘养殖',
                    '工厂化养殖', '电网降碳',
                    '捕养比', '捕捞贝藻比', '捕捞拖网', '捕捞围网',
                    '捕捞刺网', '捕捞钓业', '捕捞张网', '捕捞其他', '养殖贝占比']
    target_col = '碳净排放'
    X = numeric_df[feature_cols]
    Y = numeric_df[target_col]

    eval_metrics, feature_importances, shap_means, train_rmse_folds, val_rmse_folds, all_true, all_pred = train_and_evaluate(
        X, Y, k=5)
    print(f"平均R2分数: {eval_metrics['mean_r2']:.3f}")
    print(f"平均RMSE: {eval_metrics['mean_rmse']:.3f}")
    print(f"平均MAE: {eval_metrics['mean_mae']:.3f}")

    plot_true_vs_pred(all_true, all_pred, save_path='True_vs_Predicted_Values-RF.png')


if __name__ == '__main__':
    main()
