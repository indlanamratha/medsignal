"""Ask MedSignal: the AI agent, in the dashboard."""
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from medsignal.agent.agent import MedSignalAgent

load_dotenv()
MAX_QUESTIONS_PER_SESSION = 20
EXAMPLES = [
    "Is pancreatitis a real concern with semaglutide?",
    "What is the boxed warning for Wegovy, and how many serious semaglutide reports were there in 2025?",
    "Is optic ischaemic neuropathy a signal for semaglutide, and does the label mention it?",
    "How many tirzepatide reports were filed by consumers each year?",
]

st.set_page_config(page_title="Ask MedSignal", page_icon=":speech_balloon:", layout="wide")
st.title("Ask MedSignal")
st.caption("An AI agent that answers drug safety questions using FDA drug labels, FAERS report data, "
           "and signal statistics. Every answer shows which tools it used.")


@st.cache_resource(show_spinner="Loading search models...")
def get_agent() -> MedSignalAgent:
    return MedSignalAgent()


st.session_state.setdefault("asked", 0)
example = st.selectbox("Try an example, or type your own question below", [""] + EXAMPLES)
question = st.text_input("Your question", value=example, placeholder="e.g. Can tirzepatide cause pancreatitis?")

if st.button("Ask", type="primary", disabled=not question):
    if st.session_state.asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Refresh the page to continue.")
        st.stop()
    st.session_state.asked += 1
    with st.spinner("Thinking: searching labels and querying the data..."):
        try:
            result = get_agent().ask(question)
        except Exception as error:  # show the problem instead of a blank page
            st.error(f"Something went wrong: {error}")
            st.stop()

    st.subheader("Answer")
    st.markdown(result.answer)
    st.caption(result.disclaimer)

    with st.expander(f"How this answer was built ({len(result.steps)} tool calls)"):
        for i, step in enumerate(result.steps, start=1):
            st.markdown(f"**Step {i}: `{step['tool']}`** ({step['ms']} ms)")
            output = step["output"]
            if "error" in output:
                st.error(output["error"])
            elif step["tool"] == "search_labels":
                for p in output["passages"]:
                    st.markdown(f"- **[{p['id']}] {p['brand']}, {p['section']}**: {p['text'][:300]}...")
            elif step["tool"] == "query_faers":
                st.code(output["sql"], language="sql")
                st.dataframe(pd.DataFrame(output["rows"]), width="stretch")
            else:
                st.json(output)
