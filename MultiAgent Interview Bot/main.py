import streamlit as st
from langgraph.graph import StateGraph, END
import tempfile
import json
import os
import uuid
from pathlib import Path
from app import InterviewState, agent1_review_web_sources, agent2_resume_feedback, agent3_interview_conversation
from langchain_core.runnables import RunnableConfig

st.set_page_config(page_title="MultiAgent Interview Bot", layout="wide", page_icon="🤖")
st.title("🤖 Multi-Agent Interview Bot")
st.markdown("""
This Bot is simulates an AI-Driven interview process using multiple agents.
1. Agent 1: Analyzes job description, interviewer linked profile, and company website.
2. Agent 2: Reviews your resume.
3. Agent 3: Conducts an interactive Mock Interview with questions and feedback.""")

#User Inputs
with st.form("Interview Form"):
    job_description  = st.text_area("Job Description")
    interviewer_url = st.text_input("Interviewer's LinkedIn URL")
    company_url = st.text_input("Company Website URL")
    resume_file_path = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
    submitted = st.form_submit_button("Start Interview")


# Initialize chat history
if "current_state" not in st.session_state:
    st.session_state.current_state = None

if submitted and resume_file_path:
    #Save uploaded reusme to the tempt file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir/f"{uuid.uuid4()}_{resume_file_path.name}"
    with open(temp_path, "wb") as f:
        f.write(resume_file_path.read())

    #Initial input state
    input_state = {
        "job_description":job_description,
        "interviewer_url":interviewer_url,
        "company_url":company_url,
        "resume_file_path":str(temp_path),
        "round":1,
        "qa_chat_history":[],
    }

    #Build Graph
    graph_builder = StateGraph(InterviewState)

    graph_builder.add_node("agent1", agent1_review_web_sources)
    graph_builder.add_node("agent2", agent2_resume_feedback)
    graph_builder.add_node("agent3", agent3_interview_conversation)

    graph_builder.set_entry_point("agent1")

    graph_builder.add_edge("agent1","agent2")
    graph_builder.add_edge("agent2","agent3")
    graph_builder.add_edge("agent3","agent3") # Loop for multi-round interview

    graph_builder.add_conditional_edges("agent3", lambda x: END if x["round"] > 5 else "agent3")
    graph = graph_builder.compile()


    config = RunnableConfig()
    current_state = graph.invoke(input_state, config=config)
    st.session_state.current_state = current_state
    
if st.session_state.current_state:
    state = st.session_state.current_state
    chat_container = st.container()
    user_response = st.chat_input("Your answer")

    if user_response:
        last_question = state["qa_chat_history"][-1]["question"]
        feedback_prompt = f"""
        Interviewer asked: {last_question}
        Candidate answered: {user_response}
        Now, give detailed evaluation and an ideal answer with explanation."""

        from app import llm
        feedback = llm.invoke(feedback_prompt).content
        state["qa_chat_history"].append({
            "question": last_question,
            "answer": user_response,
            "feedback": feedback
        })
        state["round"] += 1

    # Run Graph
    current_state = input_state
    while current_state["round"] <= 5:
        current_state = graph.invoke(current_state, config=RunnableConfig())
        last_qa = current_state["qa_chat_history"][-1]

        with chat_container:
            st.chat_message("Interviewer").markdown(f"Round {current_state['round'] - 1}**: {last_qa['question']}")
            st.chat_message("You").markdown(last_qa['answer'])
            st.chat_message("Interviewer Feedback").markdown(last_qa['feedback'])

    st.success("Interview Session Completed!")

    #Save transcript
    transcript_json = json.dumps(current_state["qa_chat_history"], indent=2)
    st.download_button(
        label="📄 Download Interview Transcript",
        data=transcript_json,
        file_name="interview_transcript.json",
        type="application/json"
    )
