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
import re

# ========== 环境检测（云端自动禁用 OCR）==========
IN_CLOUD = (os.path.exists('/mount/src') or 
            os.environ.get('STREAMLIT_CLOUD', '').lower() == 'true' or
            'STREAMLIT_SHARING_MODE' in os.environ)

# ========== 页面配置 ==========
st.set_page_config(page_title="余悦错题本", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f9f7f3; }
    .stButton > button { background-color: #c8e7d5; color: #2c5f2d; border-radius: 15px; font-weight: bold; }
    .question-card { background-color: white; border-radius: 20px; padding: 20px; margin-bottom: 20px; border-left: 8px solid #ffb347; }
    .solve-card {
        background: linear-gradient(135deg, #e8f4f8 0%, #d1e9f0 100%);
        border-radius: 20px;
        padding: 20px;
        margin-top: 20px;
        border-left: 8px solid #2c5f2d;
    }
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 30px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .welcome-title { font-size: 3rem; font-weight: bold; margin-bottom: 0.5rem; }
    .welcome-subtitle { font-size: 1.3rem; opacity: 0.9; }
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
        created_at TIMESTAMP,
        solution TEXT DEFAULT '')''')
    c.execute("PRAGMA table_info(wrong_questions)")
    columns = [col[1] for col in c.fetchall()]
    if "solution" not in columns:
        c.execute('ALTER TABLE wrong_questions ADD COLUMN solution TEXT DEFAULT ""')
    conn.commit()
    conn.close()

def add_question(question_text, wrong_answer, correct_answer, knowledge_point, error_type, image_path, solution=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO wrong_questions 
        (question_text, wrong_answer, correct_answer, knowledge_point, error_type, image_path, created_at, solution) 
        VALUES (?,?,?,?,?,?,?,?)''',
              (question_text, wrong_answer, correct_answer, knowledge_point, error_type, image_path, datetime.datetime.now(), solution))
    conn.commit()
    conn.close()

def update_solution(qid, solution):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE wrong_questions SET solution = ? WHERE id = ?', (solution, qid))
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

# ========== AI 调用 ==========
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

# ========== OCR 识别（云端使用腾讯云，本地使用 EasyOCR）==========
IN_CLOUD = (os.path.exists('/mount/src') or 
            os.environ.get('STREAMLIT_CLOUD', '').lower() == 'true' or
            'STREAMLIT_SHARING_MODE' in os.environ)

def get_tencent_ocr_client():
    """使用 Streamlit Cloud 的 secrets 获取密钥并初始化客户端"""
    try:
        print("正在尝试从 st.secrets 读取密钥...")
        secret_id = st.secrets["TENCENTCLOUD_SECRET_ID"]
        secret_key = st.secrets["TENCENTCLOUD_SECRET_KEY"]
        print("成功读取密钥，开始初始化客户端...")
        cred = credential.Credential(secret_id, secret_key)
        client = ocr_client.OcrClient(cred, "ap-guangzhou")
        print("客户端初始化成功！")
        return client
    except Exception as e:
        print(f"初始化失败，错误详情: {e}")
        st.warning(f"腾讯云 OCR 初始化失败: {e}")
        return None

def recognize_text_from_image(image_bytes):
    # 云端环境使用腾讯云 OCR
    if IN_CLOUD:
        try:
            client = get_tencent_ocr_client()
            if client is None:
                return "【腾讯云 OCR 初始化失败，请检查密钥配置】"

            # 构建请求
            req = models.GeneralBasicOCRRequest()
            # 将图片转为 Base64 编码
            img_base64 = base64.b64encode(image_bytes).decode()
            req.ImageBase64 = img_base64
            
            # 可选：如需识别手写体，可取消下面一行的注释
            # req.EnableWordPolygon = True

            # 调用 API
            resp = client.GeneralBasicOCR(req)
            
            if resp.TextDetections:
                # 将每行识别结果用换行符连接
                texts = [item.DetectedText for item in resp.TextDetections]
                return "\n".join(texts)
            else:
                return "未识别到任何文字"
                
        except Exception as e:
            # 详细错误信息会打印到日志，页面上只显示友好提示
            print(f"腾讯云 OCR 调用失败: {e}")
            return "【腾讯云 OCR 调用失败，请稍后重试】"
    
    # 本地环境使用 EasyOCR
    else:
        reader = load_easyocr()
        if reader is None:
            return "【OCR 服务不可用，请手动输入题目】"
        try:
            result = reader.readtext(image_bytes)
            if not result:
                return "未识别到文字"
            texts = [item[1] for item in result if item[1].strip()]
            return "\n".join(texts) if texts else "未识别到有效文字"
        except Exception as e:
            return f"识别出错：{e}，请手动输入"

# ========== AI 解题函数 ==========
def solve_math_problem(question_text, wrong_answer=None, error_type=None):
    if not question_text:
        return "请先输入题目内容"
    wrong_hint = f"\n注意：学生的错误答案是「{wrong_answer}」，请针对这个错误进行重点讲解。" if wrong_answer else ""
    error_hint = f"\n学生的错误原因：{error_type}，请针对这个原因给出改进建议。" if error_type else ""
    prompt = f"""你是一位耐心、亲切的小学数学老师。请为下面这道数学题提供详细的解题思路和步骤。

题目：{question_text}{wrong_hint}{error_hint}

要求：
1. 用小学生能理解的语言讲解，分步骤说明
2. 先分析题目考查的知识点
3. 然后给出正确的解题步骤
4. 最后总结易错点和解题技巧
5. 整体语气要鼓励、积极

请按以下格式输出：
📚 **知识点**：...
💡 **解题思路**：...
📝 **详细步骤**：...
⚠️ **易错提醒**：...
🎯 **总结**：...
"""
    messages = [{"role": "user", "content": prompt}]
    result = call_deepseek_chat(messages, temperature=0.5)
    return result if result else "AI 解题服务暂时不可用，请稍后再试。"

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
        _, qtext, wans, cans, kp, err, _, ct, _ = q
        review += f"**题目**: {qtext}\n**错误答案**: {wans}\n**正确答案**: {cans}\n**知识点**: {kp} | **错误原因**: {err}\n\n"
    return review

def generate_two_week_paper(questions_two_weeks):
    if len(questions_two_weeks)==0: return "过去两周无错题"
    qlist = "\n".join([f"{i+1}. {q[1]}" for i,q in enumerate(questions_two_weeks)])
    prompt = f"根据以下错题，生成一套10道题的综合练习卷（含答案）。要求：题目难度适中，题型多样，最后附上答案。\n{qlist}"
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

def convert_to_printable_html(paper_text, title="综合练习卷"):
    html_content = paper_text
    html_content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'^\d+\.\s+(.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^- (.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ul>{m.group(0)}</ul>', html_content)
    lines = html_content.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('<'):
            processed_lines.append(f'<p>{line}</p>')
        elif line:
            processed_lines.append(line)
    html_content = '\n'.join(processed_lines)
    return f'''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>{title}</title>
<style>
    @media print {{ body {{ margin: 2cm; font-size: 12pt; }} .no-print {{ display: none; }} }}
    @media screen {{ body {{ max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
    .paper {{ background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }} }}
    body {{ font-family: "Microsoft YaHei", "SimHei", sans-serif; line-height: 1.6; color: #333; }}
    h1 {{ text-align: center; color: #2c5f2d; }} h2 {{ color: #2c5f2d; border-bottom: 2px solid #c8e7d5; }}
    button {{ background-color: #2c5f2d; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
</style>
</head>
<body>
<div class="paper">{html_content}</div>
<div class="no-print" style="text-align:center; margin-top:20px;">
<button onclick="window.print();">🖨️ 打印试卷</button>
<button onclick="window.close();" style="margin-left:10px;">关闭窗口</button>
</div>
</body>
</html>'''

# ========== 界面函数 ==========
def show_entry():
    st.header("📝 录入错题")

    # 初始化 session_state
    if "ocr_result" not in st.session_state:
        st.session_state.ocr_result = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_file" not in st.session_state:
        st.session_state.last_file = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "ai_solution" not in st.session_state:
        st.session_state.ai_solution = ""
    if "custom_error" not in st.session_state:
        st.session_state.custom_error = ""

    uploaded_file = st.file_uploader("🖼️ 上传题目图片", type=["jpg","jpeg","png"])

    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file:
            st.image(uploaded_file, width=300)
            if st.session_state.last_file != uploaded_file.name:
                st.session_state.ocr_result = ""
                st.session_state.last_file = uploaded_file.name
            
            if st.button("🔍 识别图片文字", disabled=st.session_state.processing):
                st.session_state.processing = True
                with st.spinner("识别中，请稍候..."):
                    raw = recognize_text_from_image(uploaded_file.getvalue())
                    if raw and "未识别" not in raw and "OCR 服务暂不可用" not in raw:
                        st.session_state.ocr_result = raw
                        st.success("✅ 识别成功！请复制下面的识别结果，粘贴到右侧编辑框中。")
                    else:
                        st.error(f"识别失败：{raw}")
                st.session_state.processing = False
                st.rerun()
            
            if st.session_state.ocr_result:
                st.markdown("**📋 识别结果（点击下方框内全选复制）**")
                st.text_area("识别文本", value=st.session_state.ocr_result, height=150, key="ocr_display")
                st.info("💡 提示：选中上面的文字，按 Ctrl+A 全选，然后 Ctrl+C 复制")

    with col2:
        st.subheader("✏️ 可编辑题目内容")
        edited = st.text_area("请在此处粘贴或手动输入题目", value=st.session_state.current_question, height=150, key="question_editor")
        st.session_state.current_question = edited

    # AI 解题窗口
    if st.session_state.current_question:
        st.markdown("---")
        st.subheader("🤖 AI 智能解题助手")
        
        col_solve1, col_solve2 = st.columns([3, 1])
        with col_solve1:
            if st.button("🎓 开始解题", key="solve_btn", use_container_width=True):
                with st.spinner("AI 老师正在思考中，请稍候..."):
                    solution = solve_math_problem(st.session_state.current_question)
                    st.session_state.ai_solution = solution
                    st.rerun()
        with col_solve2:
            if st.session_state.ai_solution:
                if st.button("📋 复制解题思路", key="copy_solution"):
                    st.success("已复制到剪贴板！")
        
        if st.session_state.ai_solution:
            st.markdown(f"""
            <div class="solve-card">
                {st.session_state.ai_solution}
            </div>
            """, unsafe_allow_html=True)

    # 题目信息输入
    wrong_answer = st.text_input("❌ 你的错误答案")
    correct_answer = st.text_input("✅ 正确答案")
    
    knowledge_options = ["四则运算", "小数意义", "三角形", "小数加减法", "观察物体", "运算定律", "统计", "数学广角", "其他"]
    knowledge_point = st.selectbox("🏷️ 知识点标签", knowledge_options)
    if knowledge_point == "其他":
        knowledge_point = st.text_input("请输入自定义知识点")
    
    # 错误原因选择
    st.markdown("**⚠️ 错误原因**")
    error_options = ["概念错误", "计算失误", "审题不清", "其他"]
    selected_error = st.radio(
        "选择错误类型",
        error_options,
        horizontal=True,
        key="error_type_radio",
        label_visibility="collapsed"
    )
    
    if selected_error == "其他":
        custom_error = st.text_input(
            "请输入具体的错误原因",
            placeholder="例如：单位换算错误、公式记错、图形画错等",
            key="custom_error_input"
        )
        st.session_state.custom_error = custom_error
        final_error_type = f"其他：{custom_error}" if custom_error else "其他（未填写具体原因）"
    else:
        final_error_type = selected_error
        st.session_state.custom_error = ""
    
    st.info(f"📌 当前错误原因：{final_error_type}")
    
    # 保存按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("📌 保存错题", key="save_btn", use_container_width=True):
            question_text = st.session_state.current_question
            
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
                add_question(question_text, wrong_answer, correct_answer, knowledge_point, final_error_type, img_path, st.session_state.ai_solution)
                st.session_state.current_question = ""
                st.session_state.ocr_result = ""
                st.session_state.ai_solution = ""
                st.session_state.last_file = None
                st.session_state.custom_error = ""
                st.success("错题已保存！")
                st.rerun()

def show_list():
    st.header("📚 我的错题本")
    questions = get_all_questions()
    if not questions:
        st.info("暂无错题，快去录入吧～")
        return
    for q in questions:
        if len(q) >= 9:
            qid, qtext, wans, cans, kp, err, img, ct, solution = q
        else:
            qid, qtext, wans, cans, kp, err, img, ct = q
            solution = ""
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
            
            if solution:
                with st.expander("📖 查看解题思路"):
                    st.markdown(solution)
            else:
                if st.button(f"🤖 AI 解题", key=f"solve_{qid}"):
                    with st.spinner("AI 老师正在解题..."):
                        new_solution = solve_math_problem(qtext, wans, err)
                        if new_solution:
                            update_solution(qid, new_solution)
                            st.success("解题思路已生成！请刷新页面查看。")
                            st.rerun()
                        else:
                            st.warning("生成失败，请稍后再试。")
            
            col1, col2 = st.columns([1, 5])
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
        for tab in [tab1, tab2, tab3, tab4]:
            with tab: st.info("暂无错题数据")
        return
    
    df = pd.DataFrame(all_questions, columns=["id","q","wa","ca","kp","err","img","ct","solution"])
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
        st.subheader("📄 两周综合练习卷")
        
        if st.button("📝 生成试卷", key="gen_paper"):
            with st.spinner("AI 正在生成试卷，请稍候..."):
                paper_text = generate_two_week_paper(two_q.to_records(index=False))
                st.session_state.generated_paper = paper_text
                st.session_state.paper_title = f"综合练习卷_{today.strftime('%Y%m%d')}"
        
        if st.session_state.get("generated_paper"):
            st.markdown("---")
            st.subheader("📖 试卷预览")
            
            with st.expander("点击展开查看试卷内容", expanded=True):
                st.markdown(st.session_state.generated_paper)
            
            st.markdown("---")
            st.subheader("📎 下载与打印")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.download_button(
                    label="📥 下载 Markdown 文件",
                    data=st.session_state.generated_paper,
                    file_name=f"{st.session_state.paper_title}.md",
                    mime="text/markdown"
                )
            
            with col_p2:
                html_content = convert_to_printable_html(
                    st.session_state.generated_paper, 
                    st.session_state.paper_title
                )
                st.download_button(
                    label="🖨️ 下载打印版 (HTML)",
                    data=html_content,
                    file_name=f"{st.session_state.paper_title}.html",
                    mime="text/html"
                )
            
            with col_p3:
                html_content = convert_to_printable_html(
                    st.session_state.generated_paper, 
                    st.session_state.paper_title
                )
                st.markdown(f"""
                <a href="data:text/html,{requests.utils.quote(html_content)}" 
                   target="_blank" 
                   style="background-color:#c8e7d5; color:#2c5f2d; padding:8px 16px; 
                          border-radius:15px; text-decoration:none; font-weight:bold;">
                    🖥️ 在新窗口打开打印版
                </a>
                """, unsafe_allow_html=True)
            
            st.info("💡 提示：点击「下载打印版」后，用浏览器打开文件，选择「文件 → 打印」或按 Ctrl+P 即可打印")
    
    with tab3:
        if st.button("生成思维导图"):
            code = generate_mind_map(all_questions)
            st.markdown(f"```mermaid\n{code}\n```")
    
    with tab4:
        if st.button("生成记忆口诀"):
            st.markdown(generate_memory_mnemonics(all_questions))

def show_welcome():
    all_q = get_all_questions()
    total_errors = len(all_q)
    today = datetime.datetime.now()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    week_errors = 0
    for q in all_q:
        ct = datetime.datetime.strptime(q[7], "%Y-%m-%d %H:%M:%S.%f")
        if ct >= start_of_week:
            week_errors += 1

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
    
    if total_errors > 0:
        kp_list = list(set([q[4] for q in all_q]))[:3]
        advice_prompt = f"余悦是一位小学四年级学生，最近数学错题涉及的知识点有：{', '.join(kp_list)}。请写一段简短、温暖、鼓励的话（不超过50字），提醒她今天可以重点复习哪些知识点。"
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

    if total_errors > 0:
        st.markdown("---")
        st.subheader("📌 最近3道错题回顾")
        recent = all_q[:3]
        for q in recent:
            qtext = q[1]
            st.markdown(f"""
            <div class="question-card" style="padding: 10px;">
                <strong>📖 {qtext[:80]}{'...' if len(qtext)>80 else ''}</strong>
            </div>
            """, unsafe_allow_html=True)

def main():
    init_db()
    st.sidebar.title("📋 功能菜单")
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
        st.sidebar.info("请在 .streamlit/secrets.toml 中设置 DEEPSEEK_API_KEY 或配置环境变量")

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