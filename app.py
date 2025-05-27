import streamlit as st
import pandas as pd
import altair as alt
import predict as pred
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'MS Gothic'
import numpy as np

# ページ設定
def page_config(title_name):
    st.set_page_config(
    page_title=title_name,
    layout='wide'
    )
    return None


# タイトル部分
def title(title_name):
    st.markdown(
        f'''
        <h1 style="text-align: center; font-size: 80px;">{title_name}</h1>
        ''',
        unsafe_allow_html = True
    )
    return None


# 入力部分
def input_data(df):
    column_list = df.columns.values.tolist()

    category_list = df[column_list[0]].unique()
    sex_list = df[column_list[1]].unique()
    age_list = sorted(df[column_list[4]].unique())
    

    c1, c2, c3, c4, c5 = st.columns(5)
    category = c1.selectbox('カテゴリーを選択', category_list)
    sex = c2.selectbox('対象性別を選択', sex_list)
    age = c3.selectbox('対象年齢を選択', age_list)
    cost = c4.number_input('投資費用(万円)を入力', min_value=0, step=10)
    index = c5.number_input('出力件数を入力', min_value=2, max_value=10, step=1)
    
    return [category, sex, age, cost], index


# 1段
def row_1(top_df, bottom_df, index):
    left, right = st.columns((1,1))

    raw_min = min(top_df['predicted_roi'].min(), bottom_df['predicted_roi'].min())
    raw_max = max(top_df['predicted_roi'].max(), bottom_df['predicted_roi'].max())
    margin = (raw_max - raw_min) * 0.1
    global_y_min = raw_min - margin
    global_y_max = raw_max + margin

    with right:
        st.subheader(f'ロケーション別 ROI分析(下位{index}件)')
        df_plot = bottom_df.nsmallest(index, 'predicted_roi').reset_index(drop=True)
        labels = [
            f'{p}-{m}月{d}-{t}時'
            for p, m, d, t in zip(df_plot['place'], df_plot['month'], df_plot['day'], df_plot['time'])
        ]
        rois = df_plot['predicted_roi']

        colors = plt.cm.plasma(np.linspace(0, 1, len(rois)))
        fig, ax = plt.subplots()

        bars = ax.bar(labels, rois - global_y_min, bottom=global_y_min, color=colors)
        ax.set_ylim(global_y_min, global_y_max)
        ax.set_xlabel("ロケーションID")
        ax.set_ylabel("広告の費用対効果 (ROI)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        y_text_pos = global_y_max - margin * 0.6
        for bar, v in zip(bars, rois):
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, y_text_pos, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

        st.pyplot(fig)

    with left:
        st.subheader(f'ロケーション別 ROI分析(上位{index}件)')
        df_plot = top_df.nlargest(index, 'predicted_roi').reset_index(drop=True)
        labels = [
            f'{p}-{m}月{d}-{t}時'
            for p, m, d, t in zip(df_plot['place'], df_plot['month'], df_plot['day'], df_plot['time'])
        ]
        rois = df_plot['predicted_roi']

        colors = plt.cm.viridis(np.linspace(0, 1, len(rois)))
        fig, ax = plt.subplots()
        bars = ax.bar(labels, rois - global_y_min, bottom=global_y_min, color=colors)
        ax.set_ylim(global_y_min, global_y_max)
        ax.set_xlabel("ロケーションID")
        ax.set_ylabel("広告の費用対効果 (ROI)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        y_text_pos = global_y_max - margin * 0.6
        for bar, v in zip(bars, rois):
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, y_text_pos, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

        st.pyplot(fig)

    st.markdown("---")
    return top_df, bottom_df, global_y_max, global_y_min, margin


# 2段目
def row_2(diff_max_df, until_100_df, index, y_max, y_min, margin):
    left, right = st.columns((1,1))

    with left:
        st.subheader(f'ロケーション別 変化最大{index}件')
        df_plot = diff_max_df.nlargest(index, 'predicted_roi').reset_index(drop=True)
        labels = [
            f'{p}-{m}月{d}-{t}時'
            for p, m, d, t in zip(df_plot['place'], df_plot['month'], df_plot['day'], df_plot['time'])
        ]
        rois = df_plot['predicted_roi']

        colors = plt.cm.viridis(np.linspace(0, 1, len(rois)))
        fig, ax = plt.subplots()
        bars = ax.bar(labels, rois - y_min, bottom=y_min, color=colors)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("ロケーションID")
        ax.set_ylabel("広告の費用対効果 (ROI)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        y_text_pos = y_max - margin * 0.6
        for bar, v in zip(bars, rois):
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, y_text_pos, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

        st.pyplot(fig)

    with right:
        st.subheader(f'損益分岐点までの{len(until_100_df)}件')
        df_plot = until_100_df.reset_index(drop=True)
        rois = df_plot['predicted_roi']

        colors = plt.cm.viridis(np.linspace(0, 1, len(rois)))
        fig, ax = plt.subplots()
        bars = ax.bar(df_plot.index, rois - y_min, bottom=y_min, color=colors)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("ロケーションID")
        ax.set_ylabel("広告の費用対効果 (ROI)")
        plt.xticks([])
        plt.tight_layout()

        st.pyplot(fig)

    st.markdown("---")
    return None


# 3段
def row_3(df, top_df, bottom_df, index):
    left, right = st.columns((1,1))

    order = ['area', 'place', 'month', 'time', 'predicted_roi']
    print(top_df.columns)
    replace_column = ['エリア', 'ロケーション', '月', '時間', '予測されるROI']
    sub_top_df = top_df[['place', 'month', 'time', 'predicted_roi']]
    sub_top_df['area'] = sub_top_df['place'].map(df.set_index('place')['area'].to_dict())
    sub_top_df = sub_top_df[order]
    for i, j in zip(order, replace_column):
        sub_top_df.rename(columns={i: j}, inplace=True)

    sub_bottom_df = bottom_df[['place', 'month', 'time', 'predicted_roi']]
    sub_bottom_df['area'] = sub_bottom_df['place'].map(df.set_index('place')['area'].to_dict())
    sub_bottom_df = sub_bottom_df[order]
    for i, j in zip(order, replace_column):
        sub_bottom_df.rename(columns={i: j}, inplace=True)

    with left:
        st.subheader(f'ロケーション別 ROI分析(上位{index}件)')
        st.dataframe(sub_top_df, use_container_width=True)

    with right:
        st.subheader(f'ロケーション別 ROI分析(下位{index}件)')
        st.dataframe(sub_bottom_df, use_container_width=True)

    return None


# 4段
def row_4(df, diff_max_df, until_100_df):
    left, right = st.columns((1,1))

    sub_top_df = diff_max_df[['place', 'month', 'time', 'predicted_roi']].copy()
    sub_top_df['area'] = sub_top_df['place'].map(df.set_index('place')['area'].to_dict())
    sub_top_df = sub_top_df[['area', 'place', 'month', 'time', 'predicted_roi']]
    sub_top_df.columns = ['エリア','ロケーション','月','時間','予測されるROI']

    sub_bottom_df = until_100_df[['place', 'month', 'time', 'predicted_roi']].copy()
    sub_bottom_df['area'] = sub_bottom_df['place'].map(df.set_index('place')['area'].to_dict())
    sub_bottom_df = sub_bottom_df[['area', 'place', 'month', 'time', 'predicted_roi']]
    sub_bottom_df.columns = ['エリア','ロケーション','月','時間','予測されるROI']

    with left:
        st.subheader('変化最大')
        st.dataframe(sub_top_df, use_container_width=True)
    with right:
        st.subheader('損益分岐点まで')
        st.dataframe(sub_bottom_df, use_container_width=True)

    return None


def main():
    title_name = 'StreetWise(プロトタイプ版)'
    path = 'all.csv'
    model_path = 'best_model.cbm'
    model = pred.loading_model(model_path)

    df = pred.replace_column(pred.load_data(path))

    page_config(title_name)
    title(title_name)
    X, index = input_data(df)
    top_df, bottom_df, diff_max_df, until_100_df = pred.place_predict(df, model, X, index)
    print(until_100_df.columns)
    top_df, bottom_df, y_max, y_min, margin = row_1(top_df, bottom_df, index)
    row_2(diff_max_df, until_100_df, index, y_max, y_min, margin)
    row_3(df, top_df, bottom_df, index)
    row_4(df, diff_max_df, until_100_df)


if __name__ == '__main__':
    main()