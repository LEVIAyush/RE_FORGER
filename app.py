import glob
import os

import streamlit as st

from agents.research_agent import ResearchAgent

st.set_page_config(
    page_title="ResearchForge",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stDecoration"] {
        display: none;
    }
    .block-container {
        padding: 0.75rem clamp(1rem, 5vw, 4rem) 2rem;
    }
    div[data-testid="stMarkdownContainer"] h2 {
        margin-top: 0;
    }
    .st-key-chat_history_container,
    .st-key-chat_history_container [data-testid="stVerticalBlockBorderWrapper"] {
        height: clamp(180px, calc(100vh - 390px), 600px) !important;
        min-height: 0 !important;
    }
    @media (max-width: 800px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
            gap: 1.5rem;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0;
        }
        .block-container {
            padding-right: 1rem;
            padding-left: 1rem;
        }
    }
    @media (max-width: 480px) {
        [data-testid="stTabs"] [data-baseweb="tab"] {
            padding-right: 0.5rem;
            padding-left: 0.5rem;
            font-size: 0.85rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🔍 ResearchForge")
st.caption("Autonomous Research Assistant — plan, search, calculate, reflect, remember.")

# ---------------- Init ----------------
if "agent" not in st.session_state:
    st.session_state.agent = ResearchAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = st.session_state.agent

left, right = st.columns([1.3, 1.2], gap="large")

# ---------------- LEFT: PROCESS TABS ----------------
with left:
    st.subheader("Agent process")

    tab_plan, tab_exec, tab_mem, tab_draft, tab_reflect, tab_final, tab_reports = st.tabs(
        ["Plan", "Execution", "Memory", "Draft", "Reflection", "Final answer", "Reports"]
    )

    plan_slot = tab_plan.empty()
    exec_slot = tab_exec.empty()
    mem_slot = tab_mem.empty()
    draft_slot = tab_draft.empty()
    reflect_slot = tab_reflect.empty()
    final_slot = tab_final.empty()

    def render_plan(steps):
        with plan_slot.container():
            if steps:
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**{i}.** {step}")
            else:
                st.info("No plan yet.")

    def render_execution(items):
        with exec_slot.container():
            if not items:
                st.info("No execution yet.")
                return
            for i, item in enumerate(items, 1):
                label = f"Step {i}: {item['step'][:60]}"
                with st.expander(label, expanded=(i == len(items))):
                    if item.get("tool_calls"):
                        for tc in item["tool_calls"]:
                            st.caption(f"🔧 {tc['tool']}({tc['input']})")
                    else:
                        st.caption("No tool used — answered directly.")
                    st.write(item["result"])

    def render_memory(recalled, items):
        with mem_slot.container():
            tool_calls_made = [tc for item in items for tc in item.get("tool_calls", [])]

            if tool_calls_made:
                st.markdown("**Tool calls this run**")
                for tc in tool_calls_made:
                    st.markdown(f"- `{tc['tool']}({tc['input']})` → {str(tc['output'])[:150]}")

            if recalled:
                st.markdown("**Relevant prior research (persisted across sessions)**")
                for r in recalled:
                    st.markdown(f"- *{r['step']}* → {r['result'][:150]}")

            if not tool_calls_made and not recalled:
                st.info("Memory is empty. It fills in as the agent uses tools and "
                         "recalls related past research.")

    def render_text_slot(slot, text, empty_message):
        with slot.container():
            if text:
                st.markdown(text)
            else:
                st.info(empty_message)

    # Initial render from whatever state the agent already holds (so a
    # st.rerun() after a question doesn't lose what was just shown).
    render_plan(agent.last_plan)
    render_execution(agent.last_steps_results)
    render_memory(agent.last_recalled_memory, agent.last_steps_results)
    render_text_slot(draft_slot, agent.last_draft, "No draft yet.")
    render_text_slot(reflect_slot, agent.last_reflection, "No reflection yet.")
    render_text_slot(final_slot, agent.last_final_answer, "No final answer yet.")

    if agent.last_final_answer:
        with tab_final:
            st.download_button(
                "Download this report (.md)",
                agent.last_report_content or agent.last_final_answer,
                file_name=os.path.basename(agent.last_report_path) if agent.last_report_path else "report.md",
                mime="text/markdown",
                key="dl_current_report",
            )

    with tab_reports:
        st.caption("Previously saved research reports (stored in `data/`).")
        files = sorted(glob.glob(os.path.join("data", "report_*.md")), key=os.path.getmtime, reverse=True)
        if not files:
            st.info("No saved reports yet. Ask a question to generate one.")
        else:
            labels = [os.path.basename(f) for f in files]
            choice = st.selectbox("Choose a report", labels, key="report_choice")
            chosen_path = os.path.join("data", choice)
            try:
                with open(chosen_path, "r", encoding="utf-8") as f:
                    content = f.read()
                st.download_button(
                    "Download this report",
                    content,
                    file_name=choice,
                    mime="text/markdown",
                    key="dl_report_browser",
                )
                with st.expander("Preview", expanded=True):
                    st.markdown(content)
            except OSError as e:
                st.error(f"Could not read report: {e}")

# ---------------- RIGHT: CHAT ----------------
with right:
    st.subheader("Chat")

    chat_container = st.container(
        height=400,
        border=True,
        key="chat_history_container",
    )

    with chat_container:
        if not st.session_state.chat_history:
            st.info("Ask a research question to start.")
        else:
            for item in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(item["question"])
                with st.chat_message("assistant"):
                    st.markdown(item["answer"])

    question = st.chat_input("Type your research question here...")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        agent.conversation_history = []
        agent.full_log = []
        agent.last_plan = []
        agent.last_steps_results = []
        agent.last_recalled_memory = []
        agent.last_draft = ""
        agent.last_reflection = ""
        agent.last_final_answer = ""
        agent.last_report_path = None
        agent.last_report_content = ""
        # Note: persistent memory_store is intentionally NOT cleared here —
        # that's the point of it being persistent. Use agent.memory_store.clear()
        # directly if you ever need to wipe it.
        st.rerun()

    if question:
        with st.spinner("Agent is researching..."):
            executed_so_far = []

            def on_event(event, payload):
                if event == "plan":
                    render_plan(payload["steps"])
                elif event == "step":
                    executed_so_far.append(
                        {
                            "step": payload["step"],
                            "result": payload["result"],
                            "tool_calls": payload["tool_calls"],
                        }
                    )
                    render_execution(executed_so_far)
                elif event == "draft":
                    render_text_slot(draft_slot, payload["text"], "No draft yet.")
                    render_memory(payload.get("recalled", []), executed_so_far)
                elif event == "reflection":
                    render_text_slot(reflect_slot, payload["text"], "No reflection yet.")
                elif event == "final":
                    render_text_slot(final_slot, payload["text"], "No final answer yet.")

            try:
                answer = agent.run(question, on_event=on_event)
                st.session_state.chat_history.append({"question": question, "answer": answer})
            except Exception as e:
                st.error(f"Something went wrong: {e}")

        st.rerun()
