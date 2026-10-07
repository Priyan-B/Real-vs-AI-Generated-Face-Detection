import glob
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
IMAGE_DIR = os.path.join(BASE_DIR, "test_images")
N_ROUNDS = 10
DISPLAY_SIZE = 512
LABELS = {1: "Real", 0: "AI-Generated"}

st.set_page_config(page_title="Spot the Fake Face", layout="wide")

# The default layout felt cramped, so I bumped up the image, buttons and text
# and added green/red result boxes so it's easy to tell who got it right.
CSS = """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.stats {display: flex; gap: 14px; margin: 6px 0 10px;}
.stat {flex: 1; padding: 12px 18px; border: 1px solid rgba(128,128,128,.35); border-radius: 12px;}
.stat .k {font-size: .95rem; opacity: .7;}
.stat .v {font-size: 2.3rem; font-weight: 800; line-height: 1.15;}
.bar {height: 10px; border-radius: 6px; background: rgba(128,128,128,.25); overflow: hidden; margin-bottom: 18px;}
.bar .fill {height: 100%; background: #4f46e5;}
div.stButton > button {width: 100%; min-height: 70px;}
div.stButton > button p {font-size: 1.35rem; font-weight: 700;}
.prompt {font-size: 1.6rem; font-weight: 800; margin: 4px 0 10px;}
.answer {font-size: 1.9rem; font-weight: 800; margin: 4px 0 8px;}
.banner {padding: 12px 16px; border-radius: 10px; font-size: 1.15rem; font-weight: 600; margin: 8px 0;}
.ok {background: rgba(22,163,74,.18); border: 1px solid #16a34a;}
.bad {background: rgba(220,38,38,.18); border: 1px solid #dc2626;}
.score {font-size: .95rem; opacity: .75; margin-bottom: 10px;}
.verdict {font-size: 1.7rem; font-weight: 800; margin-top: 14px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_pool():
    model = tf.keras.models.load_model(MODEL_PATH)
    items = []
    for folder, label in (("real", 1), ("ai", 0)):
        for path in sorted(glob.glob(os.path.join(IMAGE_DIR, folder, "*.png"))):
            img = Image.open(path).convert("RGB").resize((128, 128))
            big = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)  # display only
            items.append({"image": np.asarray(img, dtype=np.uint8),
                          "display": np.asarray(big, dtype=np.uint8), "label": label})
    batch = np.stack([it["image"] for it in items]).astype("float32") / 255.0
    probs = model.predict(batch, verbose=0).ravel()  # predict once up front
    for it, p in zip(items, probs):
        it["prob_real"] = float(p)
    return items





POOL = load_pool()

def new_game():
    st.session_state.game = {
        "order": random.sample(range(len(POOL)), k=min(N_ROUNDS, len(POOL))),
        "pos": 0, "played": 0, "human": 0, "model": 0, "answered": False,
        "history": [], "feedback": '<div class="prompt">Real or AI? Make your call.</div>'}


def pct(c, n):
    return f"{100 * c / n:.0f}%" if n else "0%"


def game_over(g):
    return g["answered"] and g["played"] >= len(g["order"])


def stats_html(g):
    n, total = g["played"], len(g["order"])
    cards = [("Round", f"{min(g['pos'] + 1, total)} / {total}"),
             ("You", f"{g['human']} / {n}"),
             ("Model", f"{g['model']} / {n}")]
    cells = "".join(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards)
    return f'<div class="stats">{cells}</div><div class="bar"><div class="fill" style="width:{100 * n / total:.0f}%"></div></div>'


def banner(ok, text):
    return f'<div class="banner {"ok" if ok else "bad"}">{text}</div>'


def accuracy_html(g):
    n = g["played"]
    cards = [("Your accuracy", pct(g["human"], n)), ("Model accuracy", pct(g["model"], n))]
    cells = "".join(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards)
    return f'<div class="stats">{cells}</div>'


def guess(choice):
    g = st.session_state.game
    if g["answered"]:
        return
    item = POOL[g["order"][g["pos"]]]
    truth = item["label"]
    model_pick = int(item["prob_real"] >= 0.5)
    conf = item["prob_real"] if model_pick == 1 else 1 - item["prob_real"]
    you_ok, model_ok = choice == truth, model_pick == truth

    g["played"] += 1
    g["human"] += int(you_ok)
    g["model"] += int(model_ok)
    g["answered"] = True
    g["history"].append({"Round": g["played"], "Answer": LABELS[truth],
                         "You": LABELS[choice], "Model": LABELS[model_pick]})

    g["feedback"] = (f'<div class="answer">It was {LABELS[truth]}</div>'
                     + banner(you_ok, f"You: {LABELS[choice]} ({'correct' if you_ok else 'wrong'})")
                     + banner(model_ok, f"Model: {LABELS[model_pick]} ({'correct' if model_ok else 'wrong'}), "
                                        f"{conf:.0%} confident")
                     + f'<div class="score">Model score P(Real) = {item["prob_real"]:.2f} (0 = AI, 1 = Real)</div>')
    if game_over(g):
        h, m = g["human"], g["model"]
        if h > m:
            verdict = f"You win, {h} to {m}."
        elif m > h:
            verdict = f"The model wins, {m} to {h}."
        else:
            verdict = f"It's a tie, {h} to {m}."
        g["feedback"] += f'<div class="verdict">{verdict}</div><div class="score">Full results are below.</div>'


def next_round():
    g = st.session_state.game
    if g["answered"] and not game_over(g):
        g["pos"] += 1
        g["answered"] = False
        g["feedback"] = '<div class="prompt">Real or AI? Make your call.</div>'


if "game" not in st.session_state:
    new_game()
g = st.session_state.game
item = POOL[g["order"][g["pos"]]]

st.title("Spot the Fake Face")
st.caption(f"You and the model each guess whether the face is a real photo or AI-generated. "
           f"Best score after {N_ROUNDS} rounds wins.")
st.markdown(stats_html(g), unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")
with left:
    st.image(item["display"])
with right:
    st.markdown(g["feedback"], unsafe_allow_html=True)
    st.button("Real", on_click=guess, args=(1,), disabled=g["answered"])
    st.button("AI-Generated", on_click=guess, args=(0,), disabled=g["answered"])
    st.button("Next Image", on_click=next_round, disabled=not g["answered"] or game_over(g))
    st.button("New Game", on_click=new_game)

if game_over(g):
    st.markdown(accuracy_html(g), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(g["history"]), hide_index=True)