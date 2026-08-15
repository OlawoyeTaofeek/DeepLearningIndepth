# pip install gradio torch transformers matplotlib --break-system-packages
#
# Run with:  python decoding_playground.py
# Then open the local URL Gradio prints (usually http://127.0.0.1:7860)

import gradio as gr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"  # swap for "gpt2-medium", "distilgpt2", etc.

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")


# ---------------------------------------------------------------------------
# Core sampling logic
# ---------------------------------------------------------------------------

def get_next_token(logits, method, temperature, top_k, top_p):
    """
    Returns (next_token_id, display_probs).
    display_probs is the FULL (unmasked, temperature-scaled) distribution,
    used purely for plotting -- so the chart shows what got excluded too.
    """
    if method == "Greedy":
        next_id = torch.argmax(logits).unsqueeze(0)
        display_probs = torch.nn.functional.softmax(logits, dim=-1)
        return next_id, display_probs

    temperature = max(float(temperature), 1e-4)
    scaled = logits / temperature
    display_probs = torch.nn.functional.softmax(scaled, dim=-1)

    working = scaled.clone()

    if top_k and top_k > 0:
        k = min(int(top_k), working.shape[-1])
        cutoff = torch.topk(working, k)[0][..., -1, None]
        working = working.masked_fill(working < cutoff, float("-inf"))

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(working, descending=True)
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_sorted = cum_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False

        remove = remove_sorted.scatter(dim=-1, index=sorted_idx, src=remove_sorted)
        working = working.masked_fill(remove, float("-inf"))

    sample_probs = torch.nn.functional.softmax(working, dim=-1)
    next_id = torch.multinomial(sample_probs, num_samples=1)
    return next_id, display_probs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(probs, chosen_id, step, method):
    top_probs, top_ids = torch.topk(probs, 15)
    tokens = [tokenizer.decode(i).strip() or "·" for i in top_ids]
    values = (top_probs.detach().numpy() * 100)
    chosen_token = tokenizer.decode(chosen_id).strip() or "·"

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    tokens_r = tokens[::-1]
    values_r = values[::-1]
    colors = cm.viridis(np.linspace(0.25, 0.95, len(values_r)))
    edge_colors = ["white" if t == chosen_token else "none" for t in tokens_r]
    lw = [2 if t == chosen_token else 0 for t in tokens_r]

    bars = ax.barh(tokens_r, values_r, color=colors, edgecolor=edge_colors, linewidth=lw)

    for bar, val, tok in zip(bars, values_r, tokens_r):
        label = f"{val:.1f}%" + ("  <- picked" if tok == chosen_token else "")
        ax.text(val + max(values_r.max() * 0.015, 0.3), bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=9.5, color="white")

    ax.set_title(f"Step {step}  ({method})", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Probability (%)", color="#c9d1d9", fontsize=10)
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#30363d")
    ax.set_xlim(0, max(values_r.max() * 1.25, 5))
    ax.grid(axis="x", color="#30363d", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Generation loop (a generator, so the UI updates live token-by-token)
# ---------------------------------------------------------------------------

def run_generation(prompt, method, temperature, top_k, top_p, max_new_tokens, seed):
    if not prompt or not prompt.strip():
        yield "Please enter a prompt.", None, gr.update(maximum=2, value=1), []
        return

    # Disable irrelevant controls depending on chosen strategy, so the
    # sliders don't silently affect a mode they shouldn't.
    eff_top_k = 0
    eff_top_p = 1.0
    if method == "Temperature":
        pass  # no truncation at all
    elif method == "Top-k":
        eff_top_k = int(top_k)
    elif method == "Top-p":
        eff_top_p = float(top_p)
    elif method == "Top-k + Top-p":
        eff_top_k = int(top_k)
        eff_top_p = float(top_p)

    torch.manual_seed(int(seed))
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    history = []
    generated_text = prompt

    for step in range(1, int(max_new_tokens) + 1):
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

        next_id, probs = get_next_token(logits, method, temperature, eff_top_k, eff_top_p)
        token_str = tokenizer.decode(next_id, skip_special_tokens=True)

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)
        generated_text += token_str

        fig = make_plot(probs, next_id.item(), step, method)
        history.append({"text": generated_text, "fig": fig})

        yield generated_text, fig, gr.update(maximum=max(len(history), 2), value=len(history)), history

        if next_id.item() == tokenizer.eos_token_id:
            break


def scrub_step(step_idx, history):
    if not history:
        return "", None
    step_idx = max(1, min(int(step_idx), len(history)))
    entry = history[step_idx - 1]
    return entry["text"], entry["fig"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Decoding Strategy Playground") as demo:
    gr.Markdown(
        "# Decoding strategy playground\n"
        "Watch autoregressive generation happen token by token, and see exactly "
        "how each decoding strategy reshapes the probability distribution at every step. "
        f"Model: `{MODEL_NAME}`."
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="The scientist looked at the data and",
                lines=3,
            )
            method = gr.Radio(
                ["Greedy", "Temperature", "Top-k", "Top-p", "Top-k + Top-p"],
                value="Temperature",
                label="Decoding strategy",
            )
            temperature = gr.Slider(0.1, 2.5, value=1.0, step=0.05,
                                     label="Temperature (ignored for Greedy)")
            top_k = gr.Slider(0, 200, value=50, step=1,
                               label="Top-k (used for Top-k / Top-k + Top-p)")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01,
                               label="Top-p (used for Top-p / Top-k + Top-p)")
            max_new_tokens = gr.Slider(1, 50, value=15, step=1, label="Max new tokens")
            seed = gr.Number(value=0, label="Random seed", precision=0)
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Generated text", lines=5, interactive=False)
            prob_plot = gr.Plot(label="Top candidate probabilities at this step")
            step_slider = gr.Slider(1, 2, value=1, step=1,
                                     label="Scrub through generation steps (after generating)")

    history_state = gr.State([])

    generate_btn.click(
        run_generation,
        inputs=[prompt, method, temperature, top_k, top_p, max_new_tokens, seed],
        outputs=[output_text, prob_plot, step_slider, history_state],
    )

    step_slider.change(
        scrub_step,
        inputs=[step_slider, history_state],
        outputs=[output_text, prob_plot],
    )

if __name__ == "__main__":
    demo.launch()