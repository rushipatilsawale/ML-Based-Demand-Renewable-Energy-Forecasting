import streamlit as st
import time

st.title("Test Auto-Reset Selectbox")

default_idx = 6
if "pick" not in st.session_state:
    st.session_state["pick"] = default_idx
if "select_time" not in st.session_state:
    st.session_state["select_time"] = time.time()

now = time.time()
elapsed = now - st.session_state["select_time"]

if st.session_state["pick"] != default_idx and elapsed >= 30:
    st.session_state["pick"] = default_idx
    st.session_state["select_time"] = now
    st.rerun()

def on_change():
    st.session_state["select_time"] = time.time()

options = [f"Hour {i}" for i in range(24)]
pick = st.selectbox("Select Hour", range(24), key="pick", on_change=on_change)

st.write(f"Currently selected: {options[pick]}")
st.write(f"Elapsed since selection: {elapsed:.1f}s")

# If non-default selection, schedule rerun at 30s
if st.session_state["pick"] != default_idx:
    remaining = max(1.0, 30.0 - elapsed)
    st.caption(f"Will auto-shift back to Live Hour in {remaining:.1f} seconds...")
    @st.fragment(run_every=remaining)
    def auto_shifter():
        if time.time() - st.session_state["select_time"] >= 30:
            st.session_state["pick"] = default_idx
            st.session_state["select_time"] = time.time()
            st.rerun()
    auto_shifter()
