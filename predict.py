import itertools
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path):
    return pd.read_csv(path, dtype={'ロケーションID':'string'})


def replace_column(df):
    columns_name = df.columns.values.tolist()
    change_name = ['category', 'sex', 'area', 'place', 'age', 'traffic', 'man', 'woman',
                '0', '10', '20', '30', '40', '50', '60', 'time', 'month', 'day',
                'roi', 'cost']
    for index, name in enumerate(columns_name):
        df.rename(columns={name: change_name[index]}, inplace=True)
    return df


def loading_model(model_path):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


def place_predict(df, model, data, index):
    places = df['place'].unique()
    times = df['time'].unique()
    months = df['month'].unique()
    days = df['day'].unique()

    combos = list(itertools.product(places, times, months, days))
    input_df = pd.DataFrame(combos, columns=['place', 'time', 'month', 'day'])

    input_df['category'] = data[0]
    input_df['sex']      = data[1]
    input_df['age']      = data[2]
    input_df['cost']     = data[3]

    cols = ['category','sex','place','age','time','month','day','cost']
    input_df = input_df[cols]

    cat_features = ['category','sex','place','time','month','day']
    for c in cat_features:
        input_df[c] = input_df[c].astype('category')

    input_df['predicted_roi'] = model.predict(input_df)

    until_100_df = input_df.sort_values('predicted_roi', ascending=False).reset_index(drop=True).copy()
    mask = until_100_df['predicted_roi'] <= 100

    if mask.any():
        first_index = mask.idxmax()
        until_100_df = until_100_df.iloc[:first_index + 1]

    top_df = input_df.sort_values('predicted_roi', ascending=False).head(index)
    bottom_df = input_df.sort_values('predicted_roi', ascending=True).head(index)

    sorted_df = input_df.sort_values('predicted_roi', ascending=False).reset_index(drop=True)
    sorted_df['diff'] = sorted_df['predicted_roi'].diff().abs()
    max_diff_index = sorted_df['diff'].idxmax()
    half = index // 2
    start_index = max(max_diff_index - half, 0)
    end_index = start_index + index
    diff_max_df = sorted_df.iloc[start_index:end_index].drop(columns='diff')

    return top_df, bottom_df, diff_max_df, until_100_df


# def place_predict(df, model, data, index):
#     places = df['place'].unique()

#     category_input = data[0]
#     sex_input      = data[1]
#     age_input      = data[2]
#     cost_input     = data[3]

#     df_place = pd.DataFrame({
#         'category': [category_input] * len(places),
#         'sex':      [sex_input]      * len(places),
#         'age':      [age_input]      * len(places),
#         'cost':     [cost_input]     * len(places),
#         'place':    places
#     })
#     df_place['predicted_roi'] = model.predict(
#         df_place[['category','sex','age','cost','place']]
#     )
#     top5_place    = df_place.nlargest(index, 'predicted_roi')
#     bottom5_place = df_place.nsmallest(index, 'predicted_roi')
#     return top5_place, bottom5_place


# def month_predict(df, model, data, index):
#     months = df['month'].unique()

#     category_input = data[0]
#     sex_input      = data[1]
#     age_input      = data[2]
#     cost_input     = data[3]
#     place_input    = data[4]

#     df_month = pd.DataFrame({
#         'category': [category_input] * len(months),
#         'sex':      [sex_input]      * len(months),
#         'age':      [age_input]      * len(months),
#         'cost':     [cost_input]     * len(months),
#         'place':    [place_input]    * len(months),
#         'month':    months
#     })
#     df_month['predicted_roi'] = model.predict(
#         df_month[['category','sex','age','cost','place','month']]
#     )
#     top5_month    = df_month.nlargest(index, 'predicted_roi')
#     bottom5_month = df_month.nsmallest(index, 'predicted_roi')
#     return top5_month, bottom5_month


# def month_place_predict(df, model, data, index):
#     months = df['month'].unique()
#     places = df['place'].unique()

#     comb   = list(itertools.product(months, places))
#     df_mpl = pd.DataFrame(comb, columns=['month','place'])
#     df_mpl['category'] = data[0]
#     df_mpl['sex']      = data[1]
#     df_mpl['age']      = data[2]
#     df_mpl['cost']     = data[3]

#     df_mpl['predicted_roi'] = model.predict(
#         df_mpl[['category','sex','age','cost','place','month']]
#     )
#     top5_mpl    = df_mpl.nlargest(index, 'predicted_roi')
#     bottom5_mpl = df_mpl.nsmallest(index, 'predicted_roi')
#     return top5_mpl, bottom5_mpl