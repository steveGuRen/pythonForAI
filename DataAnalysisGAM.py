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
from pygam import LinearGAM, s

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
    # 清洗 NaN 和 Inf 值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    Y = Y.loc[X.index]  # 确保 X 和 Y 同步

    """
    利用 GAM 进行 k 折交叉验证训练，并计算指标、特征重要性与 SHAP（替代为拟合系数）值。
    返回值：
        - eval_metrics: 字典，包含平均 R2、RMSE 和 MAE
        - feature_importances: DataFrame，使用 GAM 拟合系数表示
        - shap_means: DataFrame，占位，因 GAM 不支持 SHAP（可设为 NaN）
        - train_rmse_folds: list（GAM 无训练过程，设为空列表）
        - val_rmse_folds: list，每个 fold 验证 RMSE
        - all_true: list，所有验证集真实值
        - all_pred: list，所有验证集预测值
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []
    feature_importances = pd.DataFrame()
    val_rmse_folds = []
    all_true, all_pred = [], []
    shap_means = pd.DataFrame()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = Y.iloc[train_idx].values, Y.iloc[val_idx].values

        gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) +
                        s(6) + s(7) + s(8) + s(9) + s(10) + s(11) +
                        s(12) + s(13) + s(14)).fit(X_train, y_train)

        y_pred = gam.predict(X_val)
        all_true.extend(y_val)
        all_pred.extend(y_pred)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        val_rmse_folds.append(rmse)

        # 使用模型系数作为特征重要性近似
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(gam.coef_[1:len(X.columns) + 1]),
            'fold': fold + 1
        })
        feature_importances = pd.concat([feature_importances, fold_importance], axis=0)

        # SHAP 无法用于 GAM，这里留空或设置为 NaN
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

    # plot_correlation_heatmap(numeric_df, save_path="heatmap.png")

    # 指定模型特征与目标变量（你可根据实际需求调整列名称）
    # feature_cols = ['渔业产业GDP（万元）', '捕养比', '渔业劳动力（人）', '养殖面积/池塘公顷',
    #                 '养殖面积/工厂化立方米', '电网CO2EF',
    #                 '捕捞产量贝类', '捕捞产量藻类', '捕捞拖网占比', '捕捞围网占比',
    #                 '捕捞刺网占比', '捕捞张网占比', '捕捞钓业占比', '捕捞其他占比', '养殖贝占比']

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

    plot_true_vs_pred(all_true, all_pred, save_path='True_vs_Predicted_Values-GAM.png')


if __name__ == '__main__':
    main()
