import streamlit as st
import sqlite3
import os
import datetime
from PIL import Image
import requests
from pathlib import Path
import io
import pandas as pd
from datetime import timedelta
import numpy as np
import easyocr

# ========== 页面配置 ==========
st.set_page_config(page_title="余悦错题本", page_icon="📚", layout="wide")

# 自定义CSS（增加欢迎动画和卡片样式）
st.markdown("""
<style>
    .stApp { background-color: #f9f7f3; }
    .stButton > button { background-color: #c8e7d5; color: #2c5f2d; border-radius: 15px; font-weight: bold; }
    .question-card { background-color: white; border-radius: 20px; padding: 20px; margin-bottom: 20px; border-left: 8px solid #ffb347; }
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 30px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .welcome-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .welcome-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    .feature-card {
        background-color: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s;
    }
    .feature-card:hover { transform: translateY(-5px); }
    .feature-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .stat-number { font-size: 2rem; font-weight: bold; color: #2c5f2d; }
</style>
""", unsafe_allow_html=True)

# ========== 数据库 ==========
DB_PATH = "wrong_questions.db"
UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS wrong_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_text TEXT NOT NULL,
        wrong_answer TEXT,
        correct_answer TEXT,
        knowledge_point TEXT,
        error_type TEXT,
        image_path TEXT,
        created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def add_question(question_text, wrong_answer, correct_answer, knowledge_point, error_type, image_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO wrong_questions (question_text, wrong_answer, correct_answer, knowledge_point, error_type, image_path, created_at) VALUES (?,?,?,?,?,?,?)',
              (question_text, wrong_answer, correct_answer, knowledge_point, error_type, image_path, datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_all_questions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM wrong_questions ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return rows

def delete_question(qid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM wrong_questions WHERE id = ?', (qid,))
    conn.commit()
    conn.close()

# ========== AI 和 OCR ==========
def get_api_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except:
        return os.environ.get("DEEPSEEK_API_KEY", "")

API_KEY = get_api_key()

def call_deepseek_chat(messages, temperature=0.7):
    if not API_KEY:
        st.error("❌ 请配置 DeepSeek API Key")
        return None
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": messages, "temperature": temperature}
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

def recognize_text_from_image(image_bytes):
    try:
        reader = load_easyocr()
        result = reader.readtext(image_bytes)
        if not result:
            return "未识别到任何文字"
        texts = [item[1] for item in result if item[1].strip()]
        return "\n".join(texts) if texts else "未识别到有效文字"
    except Exception as e:
        st.error(f"OCR错误: {e}")
        return ""

# ========== 辅助函数 ==========
def generate_similar_questions(question, num=3):
    if not question: return []
    prompt = f"根据以下数学题，生成{num}道类似题（每题一行）：\n{question}"
    result = call_deepseek_chat([{"role": "user", "content": prompt}], temperature=0.8)
    return [line.strip() for line in (result or "").split('\n') if line.strip()][:num]

def generate_weekly_review(questions_week):
    if len(questions_week)==0: return "本周无错题"
    review = "## 本周错题回顾\n\n"
    for q in questions_week:
        _, qtext, wans, cans, kp, err, _, ct = q
        review += f"**题目**: {qtext}\n**错误答案**: {wans}\n**正确答案**: {cans}\n**知识点**: {kp} | **错误原因**: {err}\n\n"
    return review

def generate_two_week_paper(questions_two_weeks):
    if len(questions_two_weeks)==0: return "过去两周无错题"
    qlist = "\n".join([f"{i+1}. {q[1]}" for i,q in enumerate(questions_two_weeks)])
    prompt = f"根据以下错题，生成一套10道题的综合练习卷（含答案）：\n{qlist}"
    return call_deepseek_chat([{"role": "user", "content": prompt}], temperature=0.7) or "生成失败"

def generate_mind_map(questions):
    kps = list(set([q[4] for q in questions]))
    if not kps: return "graph TD\n  A[无数据]"
    prompt = f"将以下知识点生成Mermaid思维导图(mindmap格式): {', '.join(kps)}"
    code = call_deepseek_chat([{"role": "user", "content": prompt}], temperature=0.3)
    return code.replace("```mermaid","").replace("```","").strip() if code else "graph TD\n  A[错误]"

def generate_memory_mnemonics(questions):
    from collections import Counter
    kps = [q[4] for q in questions]
    if not kps: return "无数据"
    top = [k for k,_ in Counter(kps).most_common(3)]
    prompt = f"为知识点{', '.join(top)}编写记忆口诀（每个口诀不超过4句）"
    return call_deepseek_chat([{"role": "user", "content": prompt}], temperature=0.8) or "生成失败"

# ========== 欢迎页面 ==========
def show_welcome():
    # 获取错题总数和本周错题数（用于统计展示）
    all_q = get_all_questions()
    total_errors = len(all_q)
    # 本周错题数
    today = datetime.datetime.now()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    week_errors = 0
    for q in all_q:
        ct = datetime.datetime.strptime(q[7], "%Y-%m-%d %H:%M:%S.%f")
        if ct >= start_of_week:
            week_errors += 1

    # 欢迎卡片
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">🌟 余悦，欢迎回来！ 🌟</div>
        <div class="welcome-subtitle">你的专属AI错题本，陪你每天进步一点点</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <div class="stat-number">{total_errors}</div>
            <div>累计错题数</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">📅</div>
            <div class="stat-number">{week_errors}</div>
            <div>本周新增错题</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        # 目标进度：假设目标是95分，这里可显示一个模拟进度条
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div>目标：≥95分</div>
            <progress value="85" max="100" style="width:100%; height:10px; border-radius:5px;"></progress>
            <div>当前实力分：85</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("✨ 今日学习建议")
    
    # 调用AI生成个性化鼓励语（可选）
    if total_errors > 0:
        advice_prompt = f"余悦是一位小学四年级学生，最近数学错题涉及的知识点有：{', '.join(list(set([q[4] for q in all_q]))[:3])}。请写一段简短、温暖、鼓励的话（不超过50字），提醒她今天可以重点复习哪些知识点。"
        advice = call_deepseek_chat([{"role": "user", "content": advice_prompt}], temperature=0.7)
        if advice:
            st.info(f"💡 {advice}")
        else:
            st.info("💪 今天的努力，是明天的收获！快去看看错题本吧～")
    else:
        st.info("🎉 太棒了！目前还没有错题，继续保持！可以录入一些练习题目哦～")

    st.markdown("---")
    st.subheader("🚀 快速导航")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("📝 录入错题", use_container_width=True):
            st.session_state.menu = "录入错题"
            st.rerun()
    with col_b:
        if st.button("📖 查看错题本", use_container_width=True):
            st.session_state.menu = "错题本"
            st.rerun()
    with col_c:
        if st.button("🧠 智能复习", use_container_width=True):
            st.session_state.menu = "智能复习与总结"
            st.rerun()

    # 显示最近错题预览
    if total_errors > 0:
        st.markdown("---")
        st.subheader("📌 最近3道错题回顾")
        recent = all_q[:3]
        for q in recent:
            _, qtext, wans, cans, kp, err, _, ct = q
            st.markdown(f"""
            <div class="question-card" style="padding: 10px;">
                <strong>📖 {qtext[:80]}{'...' if len(qtext)>80 else ''}</strong><br>
                ❌ 你的答案：{wans or '未填写'} &nbsp; ✅ 正确答案：{cans}<br>
                🏷️ {kp} &nbsp; ⚠️ {err}
            </div>
            """, unsafe_allow_html=True)

# ========== 原有界面 ==========
def show_entry():
    st.header("📝 录入错题")

    if "editable_question" not in st.session_state:
        st.session_state.editable_question = ""
    if "ocr_raw" not in st.session_state:
        st.session_state.ocr_raw = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    uploaded_file = st.file_uploader("🖼️ 上传题目图片", type=["jpg","jpeg","png"])

    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file:
            st.image(uploaded_file, width=300)
            if st.session_state.last_file != uploaded_file.name:
                st.session_state.editable_question = ""
                st.session_state.ocr_raw = ""
                st.session_state.last_file = uploaded_file.name
            if st.button("🔍 识别图片文字", disabled=st.session_state.processing):
                st.session_state.processing = True
                with st.spinner("识别中，请稍候..."):
                    raw = recognize_text_from_image(uploaded_file.getvalue())
                    st.session_state.ocr_raw = raw
                    st.session_state.editable_question = raw
                st.session_state.processing = False
                st.rerun()

    with col2:
        st.subheader("✏️ 可编辑题目内容")
        edited = st.text_area("请核对并修改题目", value=st.session_state.editable_question, height=250, key="question_editor")
        st.session_state.editable_question = edited

    if st.session_state.ocr_raw:
        st.info(f"📌 原始识别结果：{st.session_state.ocr_raw[:200]}...")

    with st.form("entry_form"):
        question_text = st.session_state.editable_question
        wrong_answer = st.text_input("❌ 你的错误答案")
        correct_answer = st.text_input("✅ 正确答案")
        knowledge_options = ["四则运算", "小数意义", "三角形", "小数加减法", "观察物体", "运算定律", "统计", "数学广角", "其他"]
        knowledge_point = st.selectbox("🏷️ 知识点标签", knowledge_options)
        if knowledge_point == "其他":
            knowledge_point = st.text_input("请输入自定义知识点")
        error_type = st.radio("⚠️ 错误原因", ["概念错误", "计算失误", "审题不清"], horizontal=True)

        submitted = st.form_submit_button("📌 保存错题")
        if submitted:
            if not question_text:
                st.warning("请填写题目文本")
            elif not correct_answer:
                st.warning("请填写正确答案")
            else:
                img_path = None
                if uploaded_file:
                    img_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(img_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                add_question(question_text, wrong_answer, correct_answer, knowledge_point, error_type, img_path)
                st.session_state.editable_question = ""
                st.session_state.ocr_raw = ""
                st.session_state.last_file = None
                st.success("错题已保存！")
                st.rerun()

def show_list():
    st.header("📚 我的错题本")
    questions = get_all_questions()
    if not questions:
        st.info("暂无错题，快去录入吧～")
        return
    for q in questions:
        qid, qtext, wans, cans, kp, err, img, ct = q
        time_str = datetime.datetime.strptime(ct, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M")
        with st.container():
            st.markdown(f"""
            <div class="question-card">
                <h3>📌 错题 #{qid}</h3>
                <p><strong>题目：</strong>{qtext}</p>
                <p><strong>错误答案：</strong>{wans or '未填写'}</p>
                <p><strong>正确答案：</strong>{cans}</p>
                <p><strong>知识点：</strong>{kp} &nbsp; <strong>错误原因：</strong>{err}</p>
                <p><strong>时间：</strong>{time_str}</p>
            </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns([1,5])
            with col1:
                if st.button(f"💡 举一反三", key=f"sim_btn_{qid}"):
                    with st.spinner("生成中..."):
                        sims = generate_similar_questions(qtext)
                        if sims:
                            st.session_state[f"sim_result_{qid}"] = sims
                        else:
                            st.warning("生成失败")
            with col2:
                if st.button(f"🗑️ 删除", key=f"del_{qid}"):
                    delete_question(qid)
                    st.rerun()
            if f"sim_result_{qid}" in st.session_state:
                st.markdown("**✨ 举一反三**")
                for i, s in enumerate(st.session_state[f"sim_result_{qid}"],1):
                    st.markdown(f"{i}. {s}")
                st.markdown("---")
        st.divider()

def show_generate():
    st.header("📚 智能复习与总结")
    tab1, tab2, tab3, tab4 = st.tabs(["每周回顾", "两周综合卷", "思维导图", "记忆口诀"])
    all_questions = get_all_questions()
    if not all_questions:
        for tab in [tab1,tab2,tab3,tab4]:
            with tab: st.info("暂无错题数据")
        return
    df = pd.DataFrame(all_questions, columns=["id","q","wa","ca","kp","err","img","ct"])
    df["ct"] = pd.to_datetime(df["ct"])
    today = datetime.datetime.now()
    start_week = (today - timedelta(days=today.weekday())).replace(hour=0,minute=0,second=0)
    two_weeks_ago = today - timedelta(days=14)
    week_q = df[(df["ct"]>=start_week) & (df["ct"]<=today)]
    two_q = df[(df["ct"]>=two_weeks_ago) & (df["ct"]<=today)]
    with tab1:
        if st.button("生成本周回顾"):
            st.markdown(generate_weekly_review(week_q.to_records(index=False)))
    with tab2:
        if st.button("生成两周综合卷"):
            st.markdown(generate_two_week_paper(two_q.to_records(index=False)))
    with tab3:
        if st.button("生成思维导图"):
            code = generate_mind_map(all_questions)
            st.markdown(f"```mermaid\n{code}\n```")
    with tab4:
        if st.button("生成记忆口诀"):
            st.markdown(generate_memory_mnemonics(all_questions))

# ========== 主程序 ==========
def main():
    init_db()
    st.sidebar.title("📋 功能菜单")
    # 使用 session_state 记住当前菜单，默认为“主页”
    if "menu" not in st.session_state:
        st.session_state.menu = "主页"
    menu_options = ["主页", "录入错题", "错题本", "智能复习与总结"]
    selected = st.sidebar.radio("导航", menu_options, index=menu_options.index(st.session_state.menu))
    st.session_state.menu = selected

    st.sidebar.markdown("---")
    st.sidebar.subheader("👩‍🎓 余悦的学习助手")
    st.sidebar.write("教材：人教版数学 四年级下册")
    st.sidebar.write("目标：📈 每次考试 ≥95 分")
    if API_KEY:
        st.sidebar.success("✅ AI 助手已就绪")
    else:
        st.sidebar.error("❌ 未配置 API Key")

    if selected == "主页":
        show_welcome()
    elif selected == "录入错题":
        show_entry()
    elif selected == "错题本":
        show_list()
    else:
        show_generate()

if __name__ == "__main__":
    main()