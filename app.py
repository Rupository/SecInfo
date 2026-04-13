# [AI Disclosure: initially one-shotted by Gemini, refined by me]
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SecInfo", layout="wide")
st.title("SecInfo", text_alignment='center')
st.subheader("Topical Analysis of r/cybersecurity Posts, 2024-01-01 to 2026-04-11", text_alignment='center')

st.divider()

def clean_text(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) > 1:
            return x
        x = " ".join(map(str, x))
    
    if isinstance(x, str):
        x = x.replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
    return x

@st.cache_data
def load_data():
    full_df = pd.read_parquet('cybersec_full_data.pqt')
    topic_data = pd.read_parquet('cybersec_topic_data.pqt')
    user_data = pd.read_parquet('cybersec_user_data.pqt')

    cols_to_fix = ['Label', 'Description', 'Top (10) Keywords', 'Dominant Position', 'Supporting Arguments', 'Opposing Arguments']
    for col in cols_to_fix:
        if col in topic_data.columns:
            topic_data[col] = topic_data[col].apply(clean_text)
            
    return full_df, topic_data, user_data

full_df, topic_data, user_data = load_data()

st.header("Subreddit Overview")
col1, col2, col3, col4 = st.columns(4, border=True)
with col1:
    st.metric("Posts", len(full_df[full_df['type'] == 'post']))
with col2:
    st.metric("Comments", len(full_df[full_df['type' ]== 'comment']))
with col3:
    st.metric("Users", len(full_df['author'].unique()))
with col4:
    st.metric("Topics", len(topic_data))

st.divider()

st.header("Topic Exploration")
st.image('Topics_Over_Time.png', width='stretch')
topic_type = st.radio(label='Type',
    options=["All", "Trending", "Persistent"],
    horizontal=True,
    label_visibility='hidden'
)

filtered_topics = topic_data.copy()
if topic_type != "All":
    filtered_topics = filtered_topics[filtered_topics['Over Time'] == topic_type]

st.dataframe(
    filtered_topics[['Topic', 'Label', 'Top (10) Keywords', 'Share of Total (%)']],
    width='stretch',
    hide_index=True
)

st.divider()

st.header("Stance & Debate Analysis")
selected_topic_label = st.selectbox("Topic", topic_data['Label'], label_visibility='hidden')
topic_info = topic_data[topic_data['Label'] == selected_topic_label].iloc[0]

st.write(f"**Description:** {topic_info['Description']}")
st.info(f"**Dominant Position:** {topic_info['Dominant Position']}")

col_fav, col_opp = st.columns(2)

with col_fav:
    st.write(f"**Users in favour**: {topic_info['Favourable User Count']}")
    st.success(f"**Supporting view**: {topic_info['Arguments in Favour']}")

with col_opp:
    st.write(f"**Users against**: {topic_info['Opposed User Count']}")
    st.error(f"**Opposing view**: {topic_info['Arguments in Opposition']}")