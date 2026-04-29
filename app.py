# [AI Disclosure: initially one-shotted by Gemini, refined by me]
import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from chromadb.utils import embedding_functions

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
    full_df = pd.read_parquet('cybersec_full_data.parquet').reset_index(drop=True)
    topic_data = pd.read_parquet('cybersec_topic_data.parquet')
    
    discourse_full_df = pd.read_parquet('cybersec_discourse_full_data.parquet')
    discourse_topic_data = pd.read_parquet('cybersec_discourse_topic_data.parquet')
    
    discourse_user_data = pd.read_parquet('cybersec_discourse_user_data.parquet')

    cols_to_fix = ['Label', 'Description', 'Top (10) Keywords', 'Dominant Position', 'Supporting Arguments', 'Opposing Arguments']
    
    for df in [topic_data, discourse_topic_data]:
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
            
    return full_df, topic_data, discourse_full_df, discourse_topic_data, discourse_user_data

@st.cache_resource
def init_chroma():
    chroma_client = chromadb.PersistentClient(path="./cybersec_chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device='mps'
    )
    collection = chroma_client.get_or_create_collection(
        name="cybersecurity_rag",
        embedding_function=sentence_transformer_ef
    )
    return collection

collection = init_chroma()

def retrieve(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    enriched_contexts = []
    seen_contexts = set()
    
    if not results['documents'] or not results['documents'][0]:
        return "NO RELEVANT RETRIEVAL"
    
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        
        try:
            source_row = full_df.loc[meta['df_index']]
        except KeyError:
            continue
        
        row_type = source_row['type']
        target_link_id = source_row['link_id']
        score = source_row['score']
        date = source_row['date']
        
        if row_type == 'comment':
            parent_post = full_df[(full_df['link_id'] == target_link_id) & (full_df['type'] == 'post')]
            
            if not parent_post.empty and parent_post.iloc[0]['text'] not in seen_contexts:
                enriched_contexts.append(
                    f"RETRIEVED COMMENT [{date} | Score: {score}]: {doc_text}\n\nORIGINAL POST: {parent_post.iloc[0]['text']}"
                )
                seen_contexts.add(parent_post.iloc[0]['text'])
            else:
                enriched_contexts.append(f"RETRIEVED COMMENT [{date} | Score: {score}]: {doc_text}")
        else:
            enriched_contexts.append(f"RETRIEVED POST [{date} | Score: {score}]: {doc_text}")
            
    return "\n\n---------------------------------------------------------\n\n".join(enriched_contexts)

full_df, topic_data, discourse_full_df, discourse_topic_data, discourse_user_data = load_data()

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

st.header("Ask the Assistants!")
selected_model = st.radio(label="Choose Model:", options=["gemma4:e2b", "qwen3.5:2b"], horizontal=True, label_visibility='hidden')

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

if prompt := st.chat_input("Ask a question about the subreddit data..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching subreddit data..."):
            retrieved_context = retrieve(prompt)
            
        SYS_PROMPT = (
            "You are a cybersecurity data analyst assistant. Answer the user's question "
            "after going over the retrieved context. You may also consult the metadata (date, score) of the retrieved context. "
            "If the context does not contain the answer, try to logically deduce the answer with the retrieved context. "
            "If you are still unable to answer, explicitly state that you don't have enough information to be able to answer. "
            "Answer in one sentence preferably, at most two. Respond in the language of the user."
        )
        
        augmented_prompt = f"Context:\n{retrieved_context}\n\nQuery:\n{prompt}"
        stream = client.chat.completions.create(
            model=selected_model,
            reasoning_effort='none',
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": augmented_prompt}
            ],
            stream=True,
        )
        response = st.write_stream(stream)

        with st.expander("View Retrieved Context"):
            st.text(retrieved_context)

st.divider()

st.header("Discourse Analysis")

selected_topic_label = st.selectbox("Topic", discourse_topic_data['Label'], label_visibility='hidden')
topic_info = discourse_topic_data[discourse_topic_data['Label'] == selected_topic_label].iloc[0]

st.write(f"**Description:** {topic_info['Description']}")
st.info(f"**Dominant Position:** {topic_info['Dominant Position']}")

col_fav, col_opp = st.columns(2)

with col_fav:
    st.write(f"**Users in favour**: {topic_info['Favourable User Count']}")
    st.success(f"**Supporting view**: {topic_info['Arguments in Favour']}")

with col_opp:
    st.write(f"**Users against**: {topic_info['Opposed User Count']}")
    st.error(f"**Opposing view**: {topic_info['Arguments in Opposition']}")