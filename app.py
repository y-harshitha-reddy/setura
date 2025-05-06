# Add these imports at the top of your script with your other imports
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from serpapi import GoogleSearch 
import plotly.express as px
import re
import google.generativeai as genai
import os
import pdfplumber
import fitz
import altair as alt
from dotenv import load_dotenv
import json
import tempfile
import requests  # Already needed for NewsAPI
from datetime import datetime, timedelta  # Added for date handling

# Your existing setup
st.set_page_config(page_title="Setura", layout="wide")

# Load environment variables (already in your code)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to fetch the very latest HR-specific news
def fetch_hr_news():
    url = "https://newsapi.org/v2/everything"
    # Calculate the date 24 hours ago for the 'from' parameter
    from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "q": '"human resources" OR "HR managers" OR "human resource management" OR "HR in companies" OR "HR professionals" OR "corporate HR"',
        "apiKey": os.getenv("NEWS_API_KEY"),
        "language": "en",
        "sortBy": "publishedAt",
        "from": from_date,
        "pageSize": 10
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "ok":
            return data["articles"]
        else:
            st.error("Failed to fetch latest HR news: API error")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching latest HR news: {str(e)}")
        return []

# Update your tab structure
st.title("Setura🤝")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["JustScale", "ResumeFlow", "HireVana", "TrackVana", "HR Pulse"])

with tab1:
    st.subheader("JustScale 💰")
    st.subheader("📊 Salary Prediction & Fairness Analysis")

    # Load Data
    @st.cache_data
    def load_data():
        df = pd.read_excel(r"D:\Python project\salary_fairness_data.xlsx")
        return df

    df = load_data()

    # Encode categorical data
    le_industry = LabelEncoder()
    le_job = LabelEncoder()
    le_education = LabelEncoder()
    education_order = ["High School", "Bachelor’s", "Master’s", "PhD"]
    le_education.classes_ = np.array(education_order)
    df["Industry"] = le_industry.fit_transform(df["Industry"])
    df["Job Role"] = le_job.fit_transform(df["Job Role"])
    df["Education Level"] = df["Education Level"].apply(lambda x: education_order.index(x))

    # Features & Target
    X = df[["Years of Experience", "Industry", "Job Role", "Education Level", "Performance Score"]]
    y = df["Salary (Annual)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # User Inputs
    exp = st.slider("Years of Experience", 0, 40, 5)
    edu = st.selectbox("Education Level", education_order)
    job = st.selectbox("Job Role", le_job.classes_)
    industry = st.selectbox("Industry", le_industry.classes_)
    perf_score = st.slider("Performance Score", 1, 5, 3)
    actual_salary = st.number_input("Enter Your Salary ($)", min_value=0, value=50000)

    # Convert user input to encoded values
    edu_encoded = education_order.index(edu)
    job_encoded = le_job.transform([job])[0]
    industry_encoded = le_industry.transform([industry])[0]

    # Predict Expected Salary
    input_data = np.array([[exp, industry_encoded, job_encoded, edu_encoded, perf_score]])
    predicted_salary = model.predict(input_data)[0]

    # Calculate Fairness Score
    fairness_score = (actual_salary / predicted_salary) * 100

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Fair Salary", value=f"${predicted_salary:,.2f}")
        st.metric(label="Your Salary", value=f"${actual_salary:,.2f}")
        st.metric(label="Fairness Score", value=f"{fairness_score:.2f}%")
    with col2:
        if fairness_score < 90:
            st.error("You are underpaid! ❌")
        elif fairness_score > 110:
            st.success("You are overpaid! 💰")
        else:
            st.info("Your salary is fair ✅")

    # Visualization
    st.subheader("📈 Salary Distribution Insights")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Salary (Annual)"], bins=20, kde=True, ax=ax1)
    plt.xlabel("Salary ($)")
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["Industry"], y=df["Salary (Annual)"], ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

with tab2:
    st.subheader("ResumeFlow 📝")

    # Functions
    def get_gemini_response(input_prompt):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(input_prompt)
            if not response or not hasattr(response, 'text'):
                st.error("AI response is empty. Please try again.")
                return None
            ai_text = response.text.strip()
            json_match = re.search(r"\{.*\}", ai_text, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                return json.loads(clean_json) if isinstance(clean_json, str) else clean_json
            return ai_text
        except json.JSONDecodeError:
            st.error("Error parsing AI's JSON response. Please try again.")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None

    def extract_text_with_layout(uploaded_file):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def format_missing_keywords(missing_keywords):
        if missing_keywords:
            return "\n\nSkills: \n" + ", ".join(missing_keywords)
        return ""

    def add_keywords_to_pdf(input_pdf, missing_keywords_text):
        pdf_doc = fitz.open(input_pdf)
        last_page = pdf_doc[-1]
        text_rect = fitz.Rect(50, last_page.rect.height - 120, last_page.rect.width - 50, last_page.rect.height - 50)
        last_page.insert_textbox(text_rect, missing_keywords_text, fontsize=12, color=(0, 0, 0))
        output_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        pdf_doc.save(output_pdf)
        pdf_doc.close()
        return output_pdf

    input_prompt = """
    Hey Act Like a skilled or very experienced ATS (Application Tracking System) 
    with a deep understanding of tech field, software engineering, data science, data analysis, 
    and big data engineering. Your task is to evaluate the resume based on the given job description. 
    You must consider the job market is very competitive and you should provide 
    best assistance for improving the resumes. Assign the percentage Matching based 
    on JD and the missing keywords with high accuracy. 
    resume:{text} 
    description:{jd} 
    I want the response in one single string having the structure: 
    {{"JD Match":"%","MissingKeywords":[],"Profile Summary":""}} 
    """

    # Initialize session state
    if 'jd_history_seeker' not in st.session_state:
        st.session_state.jd_history_seeker = []
    if 'results_history_seeker' not in st.session_state:
        st.session_state.results_history_seeker = []
    if 'selected_result_seeker' not in st.session_state:
        st.session_state.selected_result_seeker = None
    if 'selected_jd_seeker' not in st.session_state:
        st.session_state.selected_jd_seeker = None
    if 'jd_history_recruiter' not in st.session_state:
        st.session_state.jd_history_recruiter = []
    if 'results_history_recruiter' not in st.session_state:
        st.session_state.results_history_recruiter = []
    if 'selected_result_recruiter' not in st.session_state:
        st.session_state.selected_result_recruiter = None
    if 'selected_jd_recruiter' not in st.session_state:
        st.session_state.selected_jd_recruiter = None

    # Sub-tabs
    tab_seeker, tab_recruiter = st.tabs(["Job Seeker", "Recruiter"])

    with tab_seeker:
        st.header("Job Seeker")
        st.text("Connecting great Resumes with great Jobs")
        jd = st.text_area("Paste the Job Description", st.session_state.selected_jd_seeker if st.session_state.selected_jd_seeker else "", key="job_seeker_jd")
        uploaded_files = [st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")]
        submit = st.button("Check➤")
        if submit:
            if uploaded_files and jd:
                if jd not in st.session_state.jd_history_seeker:
                    st.session_state.jd_history_seeker.append(jd)
                for uploaded_file in uploaded_files:
                    if uploaded_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                            temp_pdf.write(uploaded_file.read())
                            temp_pdf_path = temp_pdf.name
                        resume_text = extract_text_with_layout(temp_pdf_path)
                        response = get_gemini_response(input_prompt.format(text=resume_text, jd=jd))
                        if response:
                            try:
                                parsed_response = response if isinstance(response, dict) else json.loads(response)
                                st.session_state.results_history_seeker.append(parsed_response)
                                st.subheader(f"JD Match: {parsed_response['JD Match']}")
                                st.write(f"Profile Summary: {parsed_response['Profile Summary']}")
                                st.write(f"Missing Keywords: {parsed_response['MissingKeywords']}")
                                missing_keywords_text = format_missing_keywords(parsed_response["MissingKeywords"])
                                enhanced_pdf_filename = add_keywords_to_pdf(temp_pdf_path, missing_keywords_text)
                                with open(enhanced_pdf_filename, "rb") as file:
                                    st.download_button(
                                        label="Download Enhanced Resume PDF",
                                        data=file,
                                        file_name=f"Enhanced_{uploaded_file.name}",
                                        mime="application/pdf"
                                    )
                            except json.JSONDecodeError:
                                st.error("There was an error parsing the model's response. Please try again.")
            else:
                st.warning("Please provide both a job description and a resume.")

    with tab_recruiter:
        st.subheader("Recruiter")
        st.text("Evaluate Job Descriptions with Multiple Resumes")

with tab3:
    st.subheader("HireVana 🔍")

    # Functions
    GEMINI_API_KEY_LI = os.getenv("GEMINI_API_KEY_LI")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    genai.configure(api_key=GEMINI_API_KEY_LI)

    def search_linkedin_profiles(query):
        params = {
            "engine": "google",
            "q": f"{query} site:linkedin.com/in/",
            "num": 20,
            "api_key": SERPAPI_KEY,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        candidates = []
        if "organic_results" in results:
            for res in results["organic_results"]:
                title = res.get("title", "No Name")
                link = res.get("link", "#")
                candidates.append({"name": title, "linkedin": link})
        return candidates

    def rank_candidates_linkedin(hr_requirement, candidates):
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        candidate_texts = "\n".join([
            f"- Name: {c['name']}\n  LinkedIn: {c['linkedin']}\n  Experience: {c.get('experience', 'N/A')}\n  Skills: {', '.join(c.get('skills', []))}"
            for c in candidates
        ])
        li_prompt = f"""
        You are an AI recruitment assistant. Your job is to *evaluate and rank LinkedIn candidates* based on their experience, skills, and relevance to the job requirement.
        ### *Job Requirement:*
        {hr_requirement}
        ### *Candidates to Rank:*
        {candidate_texts}
        ### *Scoring Criteria:*
        - Give a *score between 1 and 10* (10 = Best match, 1 = Weak match).
        - *Experience*: More relevant years = Higher score.
        - *Skills Match*: More matching skills = Higher score.
        - *Role Alignment*: If past jobs match the requirement, increase score.
        *🚨 Return ONLY valid JSON in this format (no extra text):*
        ```json
        [
            {{"name": "Alice Johnson", "linkedin": "https://www.linkedin.com/in/alicejohnson", "score": 9.5}},
            {{"name": "Bob Smith", "linkedin": "https://www.linkedin.com/in/bobsmith", "score": 8.2}}
        ]
        """
        try:
            response = model.generate_content(li_prompt)
            ai_response = response.text.strip()
            json_match = re.search(r"[\s*{.}\s]", ai_response, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                ranked_candidates = json.loads(clean_json)
            else:
                raise ValueError("Invalid JSON received.")
            if not all(isinstance(c.get("score"), (int, float)) for c in ranked_candidates):
                raise ValueError("AI returned non-numeric scores.")
            return sorted(ranked_candidates, key=lambda x: x["score"], reverse=True)
        except (json.JSONDecodeError, AttributeError, ValueError):
            return [{"name": c["name"], "linkedin": c["linkedin"], "score": 5.0} for c in candidates]

    # UI
    st.sidebar.title("🔍 AI-Powered LinkedIn Hiring")
    st.sidebar.subheader("🔎 Search LinkedIn Profiles")
    search_query = st.sidebar.text_input("Enter Job Keywords (e.g., Data Scientist, Python, New York)")
    hr_requirement = st.sidebar.text_area("Describe Ideal Candidate (for LinkedIn Ranking)")
    if st.sidebar.button("🔍 Find Candidates"):
        with st.spinner("Fetching LinkedIn profiles..."):
            candidates = search_linkedin_profiles(search_query)
            if not candidates:
                st.warning("No profiles found. Try different keywords.")
            else:
                st.session_state.candidates = candidates
                st.write("### 📌 Found Candidates:")
                # --- CHANGE MADE HERE: Replaced st.write with st.markdown for clickable links ---
                for c in candidates:
                    st.markdown(f"👤 {c['name']} - [LinkedIn Profile]({c['linkedin']})")
    if st.sidebar.button("🚀 Rank LinkedIn Candidates"):
        if "candidates" not in st.session_state or not st.session_state.candidates:
            st.error("Please search for candidates first!")
        else:
            with st.spinner("Ranking candidates..."):
                ranked_candidates = rank_candidates_linkedin(hr_requirement, st.session_state.candidates)
                df = pd.DataFrame([{
                    "Name": c["name"],
                    "LinkedIn": c["linkedin"],
                    "Score": round(c["score"], 2)
                } for c in ranked_candidates])
                st.write("### 📊 LinkedIn Candidate Rankings")
                st.dataframe(df.style.format({"Score": "{:.2f}"}).highlight_max(axis=0))
                st.info("🎯 This tool fetches LinkedIn candidates via Google Search & ranks them using AI.")


with tab4:
    st.subheader("TrackVana 🚀")

    # Functions
    file_path = r"D:\Python project\recruitment_pipeline_data.xlsx"

    def load_trackvana_data(file_path):
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=["Candidate Name", "Email", "Application Date", "Current Stage", "Notes", "Phone Number"])
        return df

    def save_data(df, file_path):
        df.to_excel(file_path, index=False)
        return file_path

    # Load Data
    df_track = load_trackvana_data(file_path)

    # UI
    st.sidebar.title("🚀 Recruitment Pipeline Tracker")
    st.sidebar.header("🔍 Filters")
    stage_filter = st.sidebar.multiselect("Filter by Stage", options=df_track["Current Stage"].unique(), default=df_track["Current Stage"].unique())
    df_filtered = df_track[df_track["Current Stage"].isin(stage_filter)]
    st.markdown("#### 📋 Candidate List")
    st.dataframe(df_filtered, height=400, use_container_width=True)
    st.subheader("🛠 Update Candidate Status")
    candidate_name = st.selectbox("Select Candidate", df_track["Candidate Name"].unique())
    new_stage = st.selectbox("New Stage", ["Applied", "Shortlisted", "Interviewed", "Offered", "Rejected", "Hired"])
    notes = st.text_area("Add Notes")
    if st.button("Update Status ✅"):
        df_track.loc[df_track["Candidate Name"] == candidate_name, "Current Stage"] = new_stage
        df_track.loc[df_track["Candidate Name"] == candidate_name, "Notes"] = notes
        save_data(df_track, file_path)
        st.success(f"🎉 {candidate_name} moved to {new_stage}!")
    st.subheader("📊 Candidate Stage Distribution")
    stage_counts = df_track["Current Stage"].value_counts().reset_index()
    stage_counts.columns = ["Stage", "Count"]
    fig = px.pie(stage_counts, names="Stage", values="Count", title="Candidates at Each Stage", color_discrete_sequence=px.colors.sequential.Rainbow)
    st.plotly_chart(fig, use_container_width=True)
    if st.button("📥 Download Updated Data"):
        save_data(df_track, file_path)
        with open(file_path, "rb") as f:
            st.download_button(label="Download Excel File", data=f, file_name=os.path.basename(file_path), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.sidebar.success("🔹 Use the sidebar filters to refine the candidate list.")
# Your existing tabs (JustScale, ResumeFlow, HireVana, TrackVana) remain unchanged
# Add the new HR Pulse tab
with tab5:
    st.subheader("HR Pulse 📰")
    st.write("The Very Latest News on Human Resources and HR Management")
    
    # Fetch and display the latest HR-specific news
    with st.spinner("Fetching the very latest HR news..."):
        articles = fetch_hr_news()
    if articles:
        for article in articles:
            title = article["title"]
            url = article["url"]
            published_at = article["publishedAt"][:10]  # Show date only
            source = article["source"]["name"]
            st.markdown(f"[{title}]({url}) - {source} ({published_at})")
    else:
        st.warning("No HR news available from the last 24 hours. Check back soon!")

# Add custom CSS for minimal, professional styling
st.markdown("""
    <style>
        .faq-title {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            color: #333333;  /* Dark gray for professionalism */
            margin-bottom: 20px;
        }
        .faq-box {
            background-color: #f9f9f9;  /* Light gray for subtle background */
            padding: 10px;
            border-radius: 5px;
            box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.05);  /* Softer shadow */
            margin-bottom: 10px;
        }
        .section-title {
            font-size: 22px;
            font-weight: bold;
            color: #4A636E;  /* Muted teal for subtle emphasis */
            margin-top: 20px;
        }
    </style>
    <div class="faq-title">Frequently Asked Questions (FAQs)</div>
""", unsafe_allow_html=True)

# Function to display General Questions (to avoid repetition)
def display_general_questions():
    st.markdown('<div class="section-title">General Questions</div>', unsafe_allow_html=True)
    with st.expander("🌟 1. What is Setura?"):
        st.write("Setura is a multi-functional web app with four main tools: *JustScale* (salary prediction and fairness analysis), *ResumeFlow* (AI-powered resume analysis and job matching), *HireVana* (LinkedIn profile searching and ranking), and *TrackVana* (a recruitment pipeline tracker).")
    with st.expander("👥 2. Who can use Setura?"):
        st.write("- *Job Seekers* looking to analyze their resumes.\n- *Employers & HR professionals* for hiring and recruitment tracking.\n- *Employees* to check if their salary is fair.\n- *Recruiters* for LinkedIn candidate searches.")

# Create tabs for each app
tab1, tab2, tab3, tab4 = st.tabs(["JustScale", "ResumeFlow", "HireVana", "TrackVana"])

# JustScale Tab
with tab1:
    display_general_questions()
    st.markdown('<div class="section-title">JustScale (Salary Prediction)</div>', unsafe_allow_html=True)
    with st.expander("📈 3. How does JustScale predict salaries?"):
        st.write("It uses *machine learning* to analyze your experience, education, industry, job role, and performance score to estimate a fair salary.")
    with st.expander("⚖ 4. What is the Fairness Score?"):
        st.write("Fairness Score compares your *actual salary* to the *predicted salary:\n- **Below 90%: You might be **underpaid.\n- **90% - 110%: Your salary is **fair.\n- **Above 110%: You might be **overpaid*.")
    with st.expander("✅ 5. Can I trust the salary prediction?"):
        st.write("Yes, the model is trained on *real-world salary data, but salaries depend on various factors. This is an **estimated prediction*, not an exact value.")
    with st.expander("💵 6. Why do I need to enter my actual salary?"):
        st.write("Your salary is used to calculate *Fairness Score*, helping you understand if you are being paid fairly.")

# ResumeFlow Tab
with tab2:
    display_general_questions()
    st.markdown('<div class="section-title">ResumeFlow (AI Resume Analysis)</div>', unsafe_allow_html=True)
    with st.expander("🤖 7. How does ResumeFlow help job seekers?"):
        st.write("- It scans your *resume* and *job description.\n- It checks **matching percentage* and *missing keywords.\n- It suggests **improvements* to boost your job chances.")
    with st.expander("📄 8. What file format should I use for my resume?"):
        st.write("Only *PDF files* are supported.")
    with st.expander("⬇ 9. Can I download my improved resume?"):
        st.write("Yes, ResumeFlow *adds missing keywords* to your resume and lets you *download the enhanced version*.")
    with st.expander("📊 10. How is the JD (Job Description) match percentage calculated?"):
        st.write("The AI compares *your skills, experience, and keywords* to the job description and assigns a *match percentage*.")
    with st.expander("🔑 11. What are missing keywords?"):
        st.write("These are *important skills* or *keywords* in the job description that are missing in your resume. Adding them improves your chances of getting selected.")

# HireVana Tab
with tab3:
    display_general_questions()
    st.markdown('<div class="section-title">HireVana (LinkedIn Profile Search & Ranking)</div>', unsafe_allow_html=True)
    with st.expander("🔍 12. How does HireVana find LinkedIn profiles?"):
        st.write("It uses *Google Search API* to find *LinkedIn profiles* based on your *job title and keywords*.")
    with st.expander("🏅 13. How does it rank candidates?"):
        st.write("- AI checks *skills, experience, and job match.\n- Candidates are ranked from **1 to 10* (10 = Best Match).")
    with st.expander("❓ 14. Why are some LinkedIn profiles missing?"):
        st.write("Google might not list all LinkedIn profiles. Try different *keywords* for better results.")
    with st.expander("📬 15. Can I contact LinkedIn candidates directly?"):
        st.write("Yes, *click on the LinkedIn link* next to a candidate's name to view their profile.")

# TrackVana Tab
with tab4:
    display_general_questions()
    st.markdown('<div class="section-title">TrackVana (Recruitment Tracker)</div>', unsafe_allow_html=True)
    with st.expander("📋 16. What is TrackVana?"):
        st.write("It is a *recruitment tracking tool* for HR teams to *manage hiring stages*.")
    with st.expander("📊 17. What data can I track?"):
        st.write("- *Candidate Name\n- **Email\n- **Application Date\n- **Current Hiring Stage\n- **Notes\n- **Phone Number*")
    with st.expander("🔄 18. Can I update the candidate's hiring stage?"):
        st.write("Yes, you can *move candidates between stages* (Applied, Interviewed, Hired, etc.).")
    with st.expander("🔎 19. Can I filter candidates by stage?"):
        st.write("Yes, you can filter candidates by *different hiring stages*.")
    with st.expander("📥 20. Can I download the recruitment data?"):
        st.write("Yes, you can *export and download* the candidate list in *Excel format*.")