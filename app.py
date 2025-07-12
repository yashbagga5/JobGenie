import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    df = df.dropna(subset=['Key Skills'])
    return df

df = load_data()

# --- Logo/Photo ---
st.image('https://upload.wikimedia.org/wikipedia/commons/6/6b/Bitmap_Job.png', width=80)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stTabs [data-baseweb="tab-list"] {justify-content: center;}
    .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.title('Advanced Job Recommendation Dashboard')

# --- Sidebar: User Input ---
st.sidebar.header('Enter Your Profile')

# Multi-select dropdown for skills
all_skills = set()
for skills in df['Key Skills'].dropna():
    for skill in skills.split(','):
        skill = skill.strip()
        if skill:
            all_skills.add(skill)
all_skills = sorted(list(all_skills))
user_skills_multi = st.sidebar.multiselect('Select Your Skills', all_skills)
user_skills_text = st.sidebar.text_input('Other Skills (comma separated)', '')
user_skills = ','.join(user_skills_multi + [user_skills_text]) if user_skills_text else ','.join(user_skills_multi)

user_exp = st.sidebar.number_input('Years of Experience', min_value=0, max_value=50, value=2)
locations = ['Any'] + sorted(df['Location'].dropna().unique().tolist())
user_loc = st.sidebar.selectbox('Preferred Location', locations)
industries = ['Any'] + sorted(df['Industry'].dropna().unique().tolist())
user_ind = st.sidebar.selectbox('Preferred Industry', industries)
roles = ['Any'] + sorted(df['Role'].dropna().unique().tolist())
user_role = st.sidebar.selectbox('Preferred Role', roles)

# Inspiring quotes
quotes = [
    "Opportunities don't happen, you create them.",
    "Choose a job you love, and you will never have to work a day in your life.",
    "Success is not the key to happiness. Happiness is the key to success.",
    "Dream big and dare to fail.",
    "Your only limit is your mind."
]
st.sidebar.markdown(f"**💡 Inspiring Quote:**\n> {random.choice(quotes)}")

# --- Recommendation Engine ---
def recommend_jobs(user_skills, user_exp, user_loc, user_ind, user_role, top_n=10):
    filtered = df.copy()
    # Filter by location, industry, role
    if user_loc != 'Any':
        filtered = filtered[filtered['Location'].str.contains(user_loc, case=False, na=False)]
    if user_ind != 'Any':
        filtered = filtered[filtered['Industry'].str.contains(user_ind, case=False, na=False)]
    if user_role != 'Any':
        filtered = filtered[filtered['Role'].str.contains(user_role, case=False, na=False)]
    # Filter by experience (simple numeric match)
    if 'Job Experience Required' in filtered.columns:
        def exp_match(row):
            try:
                exp_str = str(row['Job Experience Required'])
                if '-' in exp_str:
                    min_exp = int(exp_str.split('-')[0].strip())
                else:
                    min_exp = int(''.join(filter(str.isdigit, exp_str)) or 0)
                return min_exp <= user_exp
            except:
                return True
        filtered = filtered[filtered.apply(exp_match, axis=1)]
    # TF-IDF on Key Skills
    skills_corpus = filtered['Key Skills'].fillna('').tolist()
    user_skills_str = user_skills if user_skills else ''
    tfidf = TfidfVectorizer(token_pattern=r'[^,]+')
    tfidf_matrix = tfidf.fit_transform(skills_corpus + [user_skills_str])
    user_vec = tfidf_matrix[-1]
    job_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(user_vec, job_vecs).flatten()
    filtered = filtered.copy()
    filtered['Similarity'] = sims
    top_jobs = filtered.sort_values('Similarity', ascending=False).head(top_n)
    return top_jobs

# --- Tabs Navigation ---
tabs = st.tabs(["Home", "Recommendations", "Insights", "Model Evaluation", "About"])

# --- Home Tab ---
with tabs[0]:
    st.markdown("""
    <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 20px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/Bitmap_Job.png' width='100' style='margin-right: 30px;'>
        <div>
            <h1 style='margin-bottom: 5px;'>Welcome to the Advanced Job Recommendation Dashboard!</h1>
            <h3 style='color: #4e8cff; margin-top: 0;'>Find your dream job with AI-powered recommendations</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='background: #fff; border: 2px solid #4e8cff; border-radius: 12px; box-shadow: 0 2px 8px rgba(78,140,255,0.08); padding: 24px 24px 16px 24px; margin-bottom: 24px;'>
        <h4 style='color:#2d3a4a;'>🌟 Key Features:</h4>
        <ul style='font-size:17px; color:#222; line-height:1.7;'>
            <li>🔍 <b>Smart Job Recommendations</b> tailored to your unique skills and preferences</li>
            <li>📈 <b>Interactive Job Market Analytics</b> with charts and word clouds</li>
            <li>💾 <b>Export Your Top Matches</b> for easy access</li>
            <li>💡 <b>Motivational Quotes</b> to inspire your job search journey</li>
            <li>🧠 <b>AI-Driven Skill Matching</b> for more relevant results</li>
            <li>📊 <b>Model Evaluation Tools</b> including ROC Curve and Confusion Matrix</li>
            <li>🎨 <b>Modern, User-Friendly Design</b> for a seamless experience</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.image('https://images.unsplash.com/photo-1503676382389-4809596d5290?auto=format&fit=crop&w=800&q=80', use_container_width=True, caption='Explore new opportunities!')
    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <h4>Get started by entering your profile in the sidebar and navigating through the tabs above!</h4>
        <p style='font-size:16px; margin-top: 20px;'><b>Made by Yash Bagga</b> | <a href='mailto:yashbagga5@gmail.com'>yashbagga5@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

# --- Recommendations Tab ---
with tabs[1]:
    st.header('Recommended Jobs')
    if user_skills.strip():
        recs = recommend_jobs(user_skills, user_exp, user_loc, user_ind, user_role, top_n=10)
        st.write(f"Top {len(recs)} job recommendations:")
        if not recs.empty:
            recs['Job Match Score'] = (recs['Similarity'] * 100).round(1)
            selected = st.selectbox('Select a job to see details:', recs['Job Title'] + ' @ ' + recs['Location'])
            for idx, row in recs.iterrows():
                job_label = row['Job Title'] + ' @ ' + row['Location']
                if selected == job_label:
                    st.subheader(row['Job Title'])
                    st.write(f"**Company/Industry:** {row['Industry']}")
                    st.write(f"**Location:** {row['Location']}")
                    st.write(f"**Experience Required:** {row['Job Experience Required']}")
                    st.write(f"**Role:** {row['Role']}")
                    st.write(f"**Role Category:** {row['Role Category']}")
                    st.write(f"**Key Skills:** {row['Key Skills']}")
                    st.write(f"**Salary:** {row['Job Salary']}")
                    st.write(f"**Crawl Timestamp:** {row['Crawl Timestamp']}")
                    break
            # Download button
            st.download_button('Download Recommendations (CSV)', recs.to_csv(index=False), file_name='job_recommendations.csv', mime='text/csv')
        else:
            st.warning('No jobs found matching your criteria.')
    else:
        st.info('Enter your skills in the sidebar to get recommendations.')

# --- Insights Tab ---
with tabs[2]:
    st.header('Job Market Insights')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Jobs by Location')
        top_locs = df['Location'].value_counts().head(10)
        st.bar_chart(top_locs)
    with col2:
        st.subheader('Jobs by Industry')
        top_inds = df['Industry'].value_counts().head(10)
        st.bar_chart(top_inds)
    st.subheader('Jobs by Role')
    top_roles = df['Role'].value_counts().head(10)
    st.bar_chart(top_roles)
    # Word Cloud for Key Skills
    st.subheader('Top Key Skills (Word Cloud)')
    skills_text = ','.join(df['Key Skills'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(skills_text)
    plt.figure(figsize=(10, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    # Experience Distribution
    st.subheader('Experience Required Distribution')
    def extract_min_exp(exp_str):
        try:
            if '-' in exp_str:
                return int(exp_str.split('-')[0].strip())
            return int(''.join(filter(str.isdigit, exp_str)) or 0)
        except:
            return 0
    exp_vals = df['Job Experience Required'].dropna().astype(str).apply(extract_min_exp)
    plt.figure(figsize=(8, 4))
    plt.hist(exp_vals, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Minimum Experience Required (years)')
    plt.ylabel('Number of Jobs')
    plt.title('Experience Required Distribution')
    st.pyplot(plt)

# --- Model Evaluation Tab ---
with tabs[3]:
    st.header('Model Evaluation: Industry Prediction')
    st.write('Debug: Entered Model Evaluation tab.')
    try:
        st.write('A simple classifier predicts the Industry based on Key Skills. Below are the evaluation metrics:')
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, accuracy_score
        from sklearn.preprocessing import LabelEncoder, label_binarize

        # Prepare data
        eval_df = df.dropna(subset=['Key Skills', 'Industry'])
        st.write(f'Debug: eval_df shape: {eval_df.shape}')
        X = eval_df['Key Skills']
        y = eval_df['Industry']
        tfidf_eval = TfidfVectorizer(token_pattern=r'[^,]+')
        X_tfidf = tfidf_eval.fit_transform(X)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_enc, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        # Show qualitative accuracy instead of exact value
        if acc >= 0.8:
            acc_desc = 'High'
        elif acc >= 0.6:
            acc_desc = 'Moderate'
        else:
            acc_desc = 'Low'
        st.metric('Accuracy', acc_desc)
        # Confusion Matrix (for top 5 classes)
        top_classes_idx = np.argsort(np.bincount(y_enc))[::-1][:5]
        top_classes = le.classes_[top_classes_idx]
        mask = np.isin(y_test, top_classes_idx)
        cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_classes_idx)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_classes)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        disp.plot(ax=ax_cm, xticks_rotation='vertical')
        st.pyplot(fig_cm)
        # ROC Curve (for top class)
        if len(le.classes_) > 1:
            # Binarize y for ROC
            y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
            fpr, tpr, _ = roc_curve(y_test_bin[:, top_classes_idx[0]], y_proba[:, top_classes_idx[0]])
            roc_auc = roc_auc_score(y_test_bin[:, top_classes_idx[0]], y_proba[:, top_classes_idx[0]])
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC Curve ({top_classes[0]})')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)
        else:
            st.info('ROC curve not available for single-class target.')
    except Exception as e:
        st.error(f"An error occurred in model evaluation: {e}")

# --- About Tab ---
with tabs[4]:
    st.header('About This Dashboard')
    st.write('Debug: Entered About tab.')
    try:
        st.write("""
        **Created by Yash Bagga**  
        Email: yashbagga5@gmail.com  
        This dashboard leverages advanced text similarity and interactive visualizations to help you find the best job matches.  
        Powered by Streamlit, pandas, scikit-learn, and matplotlib.
        """)
        st.image('https://images.unsplash.com/photo-1461749280684-dccba630e2f6?auto=format&fit=crop&w=800&q=80', use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred in the About tab: {e}") 