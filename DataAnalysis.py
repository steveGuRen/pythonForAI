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

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
sns.set(font="SimHei", style="white")

def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


def extract_numeric(df: pd.DataFrame) -> pd.DataFrame:

    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df


def plot_correlation_heatmap(numeric_df: pd.DataFrame, save_path: str = "heatmap.png"):

    corr = numeric_df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        linewidths=0.5
    )
    plt.title("综合参数相关性热力图")
    plt.savefig(save_path)
    plt.show()


def train_and_evaluate(X: pd.DataFrame, Y: pd.Series, k: int = 5):
    """
    利用 XGBoost 进行 k 折交叉验证训练，并计算指标、特征重要性与 SHAP 值。
    返回值：
        - eval_metrics: 字典，包含平均 R2、RMSE 和 MAE
        - feature_importances: DataFrame，包含各 fold 的特征重要性
        - shap_means: DataFrame，包含各 fold 的 SHAP 平均值
        - train_rmse_folds: list，每个 fold 训练过程的 RMSE
        - val_rmse_folds: list，每个 fold 验证过程的 RMSE
        - all_true: list，所有验证集真实值（跨 fold）
        - all_pred: list，所有验证集预测值（跨 fold）
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []
    feature_importances = pd.DataFrame()
    train_rmse_folds, val_rmse_folds = [], []
    all_true, all_pred = [], []
    shap_means = pd.DataFrame()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            eval_metric='rmse',
            random_state=42
        )

        # 模型训练
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  verbose=False)
        results = model.evals_result()

        train_rmse_folds.append(results['validation_0']['rmse'])
        val_rmse_folds.append(results['validation_1']['rmse'])

        y_pred = model.predict(X_val)
        all_true.extend(y_val.values)
        all_pred.extend(y_pred)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))

        fold_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'fold': fold + 1
        })
        feature_importances = pd.concat([feature_importances, fold_importance], axis=0)

        # 利用 SHAP 计算验证集的贡献值
        explainer = shap.Explainer(model)
        shap_values = explainer(X_val)
        fold_shap = pd.DataFrame({
            'feature': X.columns,
            'shap_mean': np.mean(shap_values.values, axis=0),
            'fold': fold + 1
        })
        shap_means = pd.concat([shap_means, fold_shap], axis=0)

    eval_metrics = {
        'mean_r2': np.mean(r2_scores),
        'mean_rmse': np.mean(rmse_scores),
        'mean_mae': np.mean(mae_scores)
    }

    return eval_metrics, feature_importances, shap_means, train_rmse_folds, val_rmse_folds, all_true, all_pred


def plot_training_process(train_rmse_folds: list, val_rmse_folds: list):

    mean_train_rmse = np.mean(train_rmse_folds, axis=0)
    mean_val_rmse = np.mean(val_rmse_folds, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_train_rmse, label='Train RMSE')
    plt.plot(mean_val_rmse, label='Validation RMSE')
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE Across Folds')
    plt.legend()
    plt.show()



def plot_feature_importance(feature_importances: pd.DataFrame, feature_columns: pd.Index,
                            save_path: str = "XGBoost_Feature_Importance.png"):

    mean_importance = feature_importances.groupby('feature')['importance'].mean()
    sorted_idx = np.argsort(mean_importance)[::-1]

    plt.figure(figsize=(16, 8))
    plt.barh(range(len(sorted_idx)), mean_importance.values[sorted_idx])
    plt.yticks(range(len(sorted_idx)), mean_importance.index[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.show()



def plot_true_vs_pred(all_true: list, all_pred: list, save_path: str = 'True_vs_Predicted_Values.png'):

    plt.figure(figsize=(10, 6))
    plt.scatter(all_true, all_pred, alpha=0.6)
    plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], '--r', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.savefig(save_path)
    plt.show()


def plot_shap_contribution(shap_means: pd.DataFrame, save_path: str = "SHAP_Contribution.png"):

    mean_shap = shap_means.groupby('feature')['shap_mean'].mean().sort_values()

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    colors = ['#FF6B6B' if val < 0 else '#4C72B0' for val in mean_shap.values]

    bars = ax.barh(mean_shap.index, mean_shap.values,
                   color=colors,
                   edgecolor='black', linewidth=0.5,
                   height=0.7, alpha=0.9)

    for bar in bars:
        width = bar.get_width()
        label_x = width if width > 0 else width - 0.02
        ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}',
            ha='left' if width < 0 else 'right',
            va='center',
            fontsize=10,
            color='black'
        )

    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    plt.title('Feature Contribution Direction (SHAP Values)', fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('Mean SHAP Value', fontsize=12, labelpad=10)
    plt.ylabel('Features', fontsize=12, labelpad=10)

    # 添加颜色图例
    import matplotlib.patches as mpatches
    positive_patch = mpatches.Patch(color='#4C72B0', label='Positive Contribution')
    negative_patch = mpatches.Patch(color='#FF6B6B', label='Negative Contribution')
    plt.legend(handles=[positive_patch, negative_patch],
               loc='upper right',
               bbox_to_anchor=(0.99, 1.05),
               frameon=True, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    positive_features = mean_shap[mean_shap > 0].index.tolist()
    negative_features = mean_shap[mean_shap < 0].index.tolist()
    print('\n正相关特征:', positive_features)
    print('负相关特征:', negative_features)



def main():
    file_path = "data/总数据- v3.xlsx"
    sheet_name = "汇总"

    df = load_data(file_path, sheet_name)
    numeric_df = extract_numeric(df)

    plot_correlation_heatmap(numeric_df, save_path="heatmap.png")

    # 指定模型特征与目标变量（你可根据实际需求调整列名称）
    feature_cols = ['渔业产业GDP（万元）', '捕养比', '渔业劳动力（人）', '养殖面积/池塘公顷',
                    '养殖面积/工厂化立方米', '电网CO2EF',
                    '捕捞产量贝类', '捕捞产量藻类', '捕捞拖网占比', '捕捞围网占比',
                    '捕捞刺网占比', '捕捞张网占比', '捕捞钓业占比', '捕捞其他占比', '养殖贝占比']
    target_col = '碳净排放'
    X = numeric_df[feature_cols]
    Y = numeric_df[target_col]

    eval_metrics, feature_importances, shap_means, train_rmse_folds, val_rmse_folds, all_true, all_pred = train_and_evaluate(
        X, Y, k=5)
    print(f"平均R2分数: {eval_metrics['mean_r2']:.3f}")
    print(f"平均RMSE: {eval_metrics['mean_rmse']:.3f}")
    print(f"平均MAE: {eval_metrics['mean_mae']:.3f}")


    plot_training_process(train_rmse_folds, val_rmse_folds)

    plot_feature_importance(feature_importances, X.columns, save_path="XGBoost_Feature_Importance.png")

    plot_true_vs_pred(all_true, all_pred, save_path='True_vs_Predicted_Values.png')

    plot_shap_contribution(shap_means, save_path="Feature_Contribution_Direction_SHAP.png")


if __name__ == '__main__':
    main()
