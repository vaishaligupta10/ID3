import streamlit as st
import pandas as pd
import numpy as np
import math

st.title("ID3 Decision Tree")

data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'High'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
                   'Yes', 'No', 'Yes', 'Yes', 'Yes',
                   'Yes', 'Yes', 'No']})

st.dataframe(data)
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return sum((-counts[i] / len(col)) * math.log2(counts[i] / len(col))
               for i in range(len(counts)))
def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted_sum = sum(
        (len(df[df[attr] == v]) / len(df)) *
        entropy(df[df[attr] == v][target])
        for v in vals)
    return total_entropy - weighted_sum
def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}
    for val in df[best].unique():
        sub_df = df[df[best] == val]
        remaining_attrs = [a for a in attrs if a != best]
        tree[best][val] = id3(sub_df, target, remaining_attrs)
    return tree
attributes = list(data.columns[:-1])
tree = id3(data, 'PlayTennis', attributes)
st.write(tree)
