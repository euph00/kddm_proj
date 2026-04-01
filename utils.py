"""
Plotting and helper utilities for CS5228 Mini Project.
Keeps the main notebook focused on analysis and narrative.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ═══════════════════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════════════════

def plot_class_balance(train):
    counts = train['Churn'].value_counts()
    counts.plot(kind='bar', color=['steelblue', 'tomato'])
    plt.xticks([0, 1], ['No Churn', 'Churn'], rotation=0)
    plt.ylabel('Count')
    plt.title('Churn Class Distribution')
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(train, num_cols):
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        for label, color in [(0, 'steelblue'), (1, 'tomato')]:
            train[train['Churn'] == label][col].hist(
                ax=axes[i], alpha=0.6, color=color, bins=30, label=str(label))
        axes[i].set_title(col, fontsize=8)
        axes[i].legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Feature Distributions by Churn (blue=No, red=Yes)', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix, title, figsize=(12, 9)):
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_point_biserial(train, num_cols):
    from scipy.stats import pointbiserialr
    correlations = {}
    for col in num_cols:
        r, _ = pointbiserialr(train['Churn'], train[col])
        correlations[col] = r
    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)

    corr_series.plot(kind='barh', figsize=(8, 5))
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Point-Biserial Correlation with Churn')
    plt.tight_layout()
    plt.show()

    top3 = corr_series.abs().nlargest(3).index.tolist()
    print(f"Top 3 features: {top3}")
    print(corr_series.round(4))
    return corr_series


def plot_service_calls_and_categorical(train):
    """Combined: service-call churn rates + International plan churn rate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cs_churn = train.groupby('Customer service calls')['Churn'].mean()
    cs_churn.plot(kind='bar', ax=axes[0], color='tomato')
    axes[0].set_title('Churn Rate by Customer Service Calls')
    axes[0].set_ylabel('Churn Rate')
    axes[0].tick_params(axis='x', rotation=0)

    intl_churn = train.groupby('International plan')['Churn'].mean()
    intl_churn.index = ['No', 'Yes']
    intl_churn.plot(kind='bar', ax=axes[1], color='tomato')
    axes[1].set_title('Churn Rate by International Plan')
    axes[1].set_ylabel('Churn Rate')
    axes[1].set_ylim(0, 0.5)
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Unsupervised Learning
# ═══════════════════════════════════════════════════════════════════════

def plot_umap_3d_pair(X_umap, color_arrays, titles, suptitle):
    """Side-by-side 3D UMAP scatter plots with two colour schemes."""
    fig = plt.figure(figsize=(14, 6))
    for idx, (colors, title) in enumerate(zip(color_arrays, titles)):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
                   c=colors, alpha=0.4, s=5)
        ax.set_title(title)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_zlabel('UMAP-3')
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_elbow_silhouette(K_range, inertias, silhouettes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(K_range), inertias, 'o-')
    axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[1].plot(list(K_range), silhouettes, 'o-', color='tomato')
    axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score')
    plt.tight_layout()
    plt.show()


def plot_cluster_profile(cluster_profile):
    plt.figure(figsize=(12, max(3, len(cluster_profile) + 1)))
    sns.heatmap(cluster_profile, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('K-Means Cluster Profiles (mean of scaled features; 0 = population average)')
    plt.tight_layout()
    plt.show()


def plot_cluster_umap_and_churn(X_umap, km_labels, cluster_names, cluster_colors,
                                train, optimal_k):
    """UMAP 3D cluster assignment + churn rate bar chart."""
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    for k in range(optimal_k):
        mask = km_labels == k
        ax1.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2],
                    c=cluster_colors[k], alpha=0.4, s=5, label=cluster_names[k])
    ax1.set_title('K-Means Cluster Assignment (UMAP 3D)')
    ax1.set_xlabel('UMAP-1'); ax1.set_ylabel('UMAP-2'); ax1.set_zlabel('UMAP-3')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    churn_by_cluster = train.assign(Cluster=km_labels).groupby('Cluster')['Churn'].mean()
    bar_colors = [cluster_colors[k] for k in churn_by_cluster.index]
    churn_by_cluster.index = [cluster_names[k] for k in churn_by_cluster.index]
    churn_by_cluster.plot(kind='bar', ax=ax2, color=bar_colors)
    ax2.axhline(train['Churn'].mean(), color='black', linestyle='--',
                linewidth=1, label='Overall mean')
    ax2.set_title('Churn Rate per K-Means Cluster')
    ax2.set_ylabel('Churn Rate')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=15)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    print("Churn rate per cluster:")
    print(churn_by_cluster.round(3))


def plot_dbscan_results(X_umap, db_labels, train, eps, min_samples):
    """k-distance inspired DBSCAN label plot + churn bar chart."""
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    print(f"DBSCAN (eps={eps}, min_samples={min_samples}): "
          f"{n_clusters} clusters, {n_noise} noise ({n_noise/len(db_labels):.1%})")

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    for label in sorted(set(db_labels)):
        mask = db_labels == label
        name = 'Noise' if label == -1 else f'Cluster {label}'
        ax1.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2],
                    alpha=0.2 if label == -1 else 0.5, s=5, label=name)
    ax1.set_title(f'DBSCAN Labels on UMAP (eps={eps}, min_samples={min_samples})')
    ax1.set_xlabel('UMAP-1'); ax1.set_ylabel('UMAP-2'); ax1.set_zlabel('UMAP-3')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    churn_db = train.assign(Label=db_labels).groupby('Label')['Churn'].mean()
    churn_db.index = ['Noise' if i == -1 else f'Cluster {i}' for i in churn_db.index]
    churn_db.plot(kind='bar', ax=ax2, color='tomato')
    ax2.axhline(train['Churn'].mean(), color='black', linestyle='--',
                linewidth=1, label='Overall mean')
    ax2.set_title('Churn Rate per DBSCAN Label')
    ax2.set_ylabel('Churn Rate')
    ax2.tick_params(axis='x', rotation=15)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Supervised Learning
# ═══════════════════════════════════════════════════════════════════════

def get_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        raw = model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    return model.predict(X)


def evaluate_split(model, X, y):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix as cm_fn,
                                 roc_auc_score, average_precision_score)
    y_pred = model.predict(X)
    y_score = get_scores(model, X)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_score),
        'pr_auc': average_precision_score(y, y_score),
        'cm': cm_fn(y, y_pred),
        'y_score': y_score,
    }


def plot_model_performance(results_df, curve_data, y_test):
    """Four-panel visualisation: metric bars, confusion matrices, ROC, PR."""
    # 1 — Metric bars
    metric_cols = ['Test Precision', 'Test Recall', 'Test F1', 'Test ROC-AUC', 'Test PR-AUC']
    plot_df = results_df[['Model'] + metric_cols].melt(
        id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(11, 5))
    sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric')
    plt.ylim(0, 1.05)
    plt.title('Model Comparison on Test Set Metrics')
    plt.ylabel('Score'); plt.xlabel('')
    plt.legend(loc='lower right')
    plt.tight_layout(); plt.show()

    # 2 — Confusion matrices
    n = len(results_df)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = np.array([axes])
    for i, row in results_df.iterrows():
        sns.heatmap(row['Test CM'], annot=True, fmt='d', cmap='Oranges',
                    cbar=False, ax=axes[i])
        axes[i].set_title(f"{row['Model']}\nTest Confusion Matrix")
        axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
    plt.tight_layout(); plt.show()

    # 3 — ROC curves
    plt.figure(figsize=(6, 5))
    for _, row in results_df.iterrows():
        d = curve_data[row['Model']]
        plt.plot(d['fpr'], d['tpr'],
                 label=f"{row['Model']} (AUC={row['Test ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title('ROC Curves (Test Set)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(); plt.tight_layout(); plt.show()

    # 4 — Precision-Recall curves
    baseline = y_test.mean()
    plt.figure(figsize=(6, 5))
    for _, row in results_df.iterrows():
        d = curve_data[row['Model']]
        plt.plot(d['pr_recall'], d['pr_precision'],
                 label=f"{row['Model']} (AP={row['Test PR-AUC']:.3f})")
    plt.axhline(baseline, color='gray', linestyle='--', linewidth=1,
                label=f'Baseline={baseline:.3f}')
    plt.title('Precision-Recall Curves (Test Set)')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(); plt.tight_layout(); plt.show()


def plot_feature_importance(best_models, num_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lr = best_models['Logistic Regression'].named_steps['model']
    lr_coef = pd.Series(lr.coef_.ravel(), index=num_cols).sort_values()
    lr_coef.plot(kind='barh', ax=axes[0],
                 color=['tomato' if v > 0 else 'steelblue' for v in lr_coef.values])
    axes[0].axvline(0, color='black', linewidth=1)
    axes[0].set_title('Logistic Regression Coefficients')

    rf = best_models['Random Forest'].named_steps['model']
    rf_imp = pd.Series(rf.feature_importances_, index=num_cols).sort_values()
    rf_imp.plot(kind='barh', ax=axes[1], color='mediumseagreen')
    axes[1].set_title('Random Forest Feature Importance')

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Generative AI (Stage 5)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_synthetic(raw_df, num_cols, scaler):
    """Apply the same preprocessing pipeline as Stage 1 to synthetic data.
    Returns (X_df, y) where X_df is a DataFrame with proper column names."""
    df = raw_df.copy()
    df['Churn'] = df['Churn'].map({'True': 1, 'False': 0, True: 1, False: 0})
    df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
    df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})
    drop = ['Total day charge', 'Total eve charge', 'Total night charge',
            'Total intl charge', 'Area code', 'State', 'Voice mail plan']
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)
    cols = [c for c in num_cols if c in df.columns]
    df[cols] = scaler.transform(df[cols])
    return df[cols], df['Churn'].values


def method_selection_table(train_orig, synth_dict):
    """Print comparison table of churn rate & categorical fidelity."""
    print(f"{'Method':<20} {'Churn Rate':>12} {'Intl Plan Yes%':>16} {'VoiceMail Yes%':>16}")
    print('=' * 66)
    real_churn = train_orig['Churn'].map({True: 1, 'True': 1, False: 0, 'False': 0}).mean()
    real_intl = (train_orig['International plan'] == 'Yes').mean()
    real_vm = (train_orig['Voice mail plan'] == 'Yes').mean()
    print(f"{'Real Training':<20} {real_churn:>11.3f} {real_intl:>15.3f} {real_vm:>15.3f}")
    print('-' * 66)
    for name, df in synth_dict.items():
        churn = df['Churn'].map({True: 1, 'True': 1, False: 0, 'False': 0}).mean()
        intl = (df['International plan'] == 'Yes').mean()
        vm = (df['Voice mail plan'] == 'Yes').mean()
        print(f"{name:<20} {churn:>11.3f} {intl:>15.3f} {vm:>15.3f}")


def plot_histogram_comparison(datasets, compare_cols, colors=None):
    """Side-by-side histograms for each column across datasets."""
    if colors is None:
        colors = {'Real Test': 'steelblue', 'TVAE Synthetic': 'mediumseagreen'}
    n_cols = len(compare_cols)
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_cols, n_ds, figsize=(6 * n_ds, 3.5 * n_cols))
    for i, col in enumerate(compare_cols):
        for j, (label, df) in enumerate(datasets.items()):
            ax = axes[i, j]
            c = colors[label]
            if df[col].dtype == 'object' or df[col].nunique() <= 5:
                df[col].value_counts().sort_index().plot(kind='bar', ax=ax,
                                                         color=c, alpha=0.7)
            else:
                ax.hist(df[col], bins=30, color=c, alpha=0.7, edgecolor='white')
            ax.set_title(f'{col} — {label} (n={len(df)})')
            ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_density_overlay(datasets, continuous_cols, colors=None):
    if colors is None:
        colors = {'Real Test': 'steelblue', 'TVAE Synthetic': 'mediumseagreen'}
    fig, axes = plt.subplots(1, len(continuous_cols),
                             figsize=(6 * len(continuous_cols), 5))
    for i, col in enumerate(continuous_cols):
        for label, df in datasets.items():
            axes[i].hist(df[col], bins=30, alpha=0.5, label=label,
                         color=colors[label], density=True, edgecolor='white')
        axes[i].set_title(col)
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=9)
    plt.suptitle('Density Overlay: Real Test vs TVAE Synthetic', y=1.02)
    plt.tight_layout()
    plt.show()


def print_summary_stats(datasets, compare_cols):
    print('Summary Statistics: Real Test vs TVAE Synthetic')
    print('=' * 70)
    for col in compare_cols:
        print(f'\n--- {col} ---')
        for label, df in datasets.items():
            if df[col].dtype == 'object':
                dist = df[col].value_counts(normalize=True).to_dict()
                print(f'  {label:18s}: {dist}')
            else:
                print(f'  {label:18s}: mean={df[col].mean():.2f}, '
                      f'std={df[col].std():.2f}, '
                      f'min={df[col].min():.2f}, max={df[col].max():.2f}')


def plot_genai_metrics(genai_results_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    bar_colors = ['steelblue', 'mediumseagreen']
    for i, metric in enumerate(metrics):
        pivot = genai_results_df.pivot(index='Model', columns='Test Set', values=metric)
        pivot = pivot[['Real Test', 'TVAE Synthetic']]
        pivot.plot(kind='bar', ax=axes[i], rot=15, color=bar_colors)
        axes[i].set_title(metric, fontsize=13)
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1)
        axes[i].legend(fontsize=11)
    plt.suptitle('Model Performance: Real Test vs TVAE Synthetic', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_genai_confusion_matrices(best_models, test_sets):
    n_models = len(best_models)
    n_sets = len(test_sets)
    fig, axes = plt.subplots(n_models, n_sets, figsize=(5 * n_sets, 4 * n_models))
    for i, (model_name, model) in enumerate(best_models.items()):
        for j, (test_name, X_eval, y_eval) in enumerate(test_sets):
            y_pred = model.predict(X_eval)
            cm = confusion_matrix(y_eval, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i, j],
                        xticklabels=['No Churn', 'Churn'],
                        yticklabels=['No Churn', 'Churn'])
            axes[i, j].set_title(f'{model_name} — {test_name}')
            axes[i, j].set_ylabel('Actual')
            axes[i, j].set_xlabel('Predicted')
    plt.tight_layout()
    plt.show()


def print_performance_gap(genai_results_df):
    real = genai_results_df[genai_results_df['Test Set'] == 'Real Test'].set_index('Model')
    tvae = genai_results_df[genai_results_df['Test Set'] == 'TVAE Synthetic'].set_index('Model')
    cols = ['Accuracy', 'Precision', 'Recall', 'F1']
    gap = tvae[cols] - real[cols]
    print('Performance Gap (TVAE Synthetic - Real Test):')
    print('=' * 70)
    print(gap.round(4).to_string())
    return gap
