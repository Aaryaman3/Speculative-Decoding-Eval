"""
Live benchmark dashboard — Streamlit app.

Two modes:
  1. LIVE DEMO   — side-by-side metrics + streaming race
  2. RESULTS     — loads from results/raw/*.jsonl, shows analysis plots
"""

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import aiohttp
import pandas as pd
import requests as req
import streamlit as st

st.set_page_config(
    page_title="EAGLE-3 Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
)

RESULTS_DIR = Path("results/raw")

GPU_HOURLY_RATES = {"T4": 0.35, "L4": 0.70, "A100": 3.67, "mac": 0.00}

st.sidebar.title("⚡ EAGLE-3 Benchmark")
mode = st.sidebar.radio("Mode", ["Live Demo", "Results Analysis"])

# ── LIVE DEMO ──────────────────────────────────────────────────────────────────
if mode == "Live Demo":
    st.title("⚡ Live: Greedy Baseline vs EAGLE-3 Speculative Decoding")
    st.caption(
        "Both servers receive the **same request** simultaneously. "
        "EAGLE-3 uses a lightweight draft model to propose 5 tokens at once; "
        "the target model verifies them in a single forward pass — "
        "reducing the number of sequential target-model calls needed."
    )

    with st.expander("ℹ️ Metric glossary", expanded=False):
        st.markdown(
            "| Metric | Meaning | Better |\n"
            "|--------|---------|--------|\n"
            "| **TPOT** | Time Per Output Token — latency between each token after the first. **The key speculative-decoding metric.** | Lower |\n"
            "| **TTFT** | Time to First Token | Lower |\n"
            "| **Tokens/sec** | Total output tokens ÷ total wall time | Higher |\n"
            "| **Acceptance rate** | % of draft tokens accepted without regeneration. ~50% → ~2× theoretical speedup | Higher |\n"
            "| **Speedup** | Baseline TPOT ÷ EAGLE-3 TPOT | Higher |"
        )

    st.sidebar.subheader("Server Config")
    baseline_url = st.sidebar.text_input("Baseline URL", "http://localhost:8000")
    eagle3_url   = st.sidebar.text_input("EAGLE-3 URL",  "http://localhost:8001")
    gpu_type     = st.sidebar.selectbox("GPU Type", ["A100", "L4", "T4", "mac"])

    st.sidebar.subheader("Request Config")
    task = st.sidebar.selectbox("Task", ["chat", "code", "summarization"])
    concurrency = st.sidebar.select_slider(
        "Concurrency", [1, 4, 8, 16, 32], value=1,
        help="Keep at 1 for clearest per-request speedup. Raise to stress-test batching.",
    )
    max_tokens = st.sidebar.slider("Max output tokens", 64, 512, 256)

    DEFAULT_PROMPTS = {
        "chat": "Explain the concept of recursion in simple terms.",
        "code": "Write a Python function that checks if a string is a palindrome.",
        "summarization": (
            "Summarize the following: The Apollo 11 mission was the first crewed lunar "
            "landing mission. Launched on July 16, 1969, it carried astronauts Neil Armstrong, "
            "Buzz Aldrin, and Michael Collins. Armstrong and Aldrin landed on the Moon on "
            "July 20, while Collins orbited above. Armstrong became the first person to walk "
            "on the Moon."
        ),
    }
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPTS[task], height=100)

    # ── helpers ────────────────────────────────────────────────────────────────

    def get_acceptance_rate(url: str) -> float | None:
        """Compute acceptance rate from vLLM counters (vLLM v0.20 removed the pre-computed metric)."""
        try:
            r = req.get(f"{url}/metrics", timeout=4)
            accepted, drafted = None, None
            for line in r.text.split("\n"):
                if line.startswith("#"):
                    continue
                if "spec_decode_num_accepted_tokens_total{" in line:
                    accepted = float(line.split()[-1])
                elif "spec_decode_num_draft_tokens_total{" in line:
                    drafted = float(line.split()[-1])
            if accepted is not None and drafted and drafted > 0:
                return accepted / drafted
        except Exception:
            pass
        return None

    async def send_one_async(session, url, model_name, prompt_text, gpu_type_str):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t_start = time.perf_counter()
        t_first = None
        text = ""
        chunk_count = 0
        server_tokens = None
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage and "completion_tokens" in usage:
                            server_tokens = usage["completion_tokens"]
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        content = choices[0]["delta"].get("content", "")
                        if content:
                            if t_first is None:
                                t_first = time.perf_counter()
                            text += content
                            chunk_count += 1
                    except Exception:
                        continue
        except Exception as e:
            return {"error": str(e), "ttft_ms": None, "tpot_ms": None,
                    "tokens_per_sec": None, "cost_usd": None, "text": ""}

        t_end = time.perf_counter()
        total = t_end - t_start
        tokens = server_tokens if server_tokens is not None else chunk_count
        ttft = (t_first - t_start) * 1000 if t_first else None
        tpot = ((t_end - t_first) / (tokens - 1) * 1000) if t_first and tokens > 1 else None
        cost = GPU_HOURLY_RATES.get(gpu_type_str, 0) / 3600 * total

        return {
            "ttft_ms": round(ttft, 1) if ttft else None,
            "tpot_ms": round(tpot, 1) if tpot else None,
            "tokens_per_sec": round(tokens / total, 1) if total > 0 else None,
            "output_tokens": tokens,
            "cost_usd": cost,
            "text": text,
            "error": None,
        }

    async def run_benchmark_async(prompt_text, concurrency_n):
        prompts = [prompt_text] * concurrency_n
        connector = aiohttp.TCPConnector(limit=concurrency_n * 2 + 4)
        async with aiohttp.ClientSession(connector=connector) as session:
            all_results = await asyncio.gather(
                *[send_one_async(session, baseline_url, "baseline", p, gpu_type) for p in prompts],
                *[send_one_async(session, eagle3_url,  "eagle3",   p, gpu_type) for p in prompts],
            )
        return list(all_results[:concurrency_n]), list(all_results[concurrency_n:])

    def avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    # ── streaming race helper ──────────────────────────────────────────────────

    def stream_sync(url: str, model_name: str, prompt_text: str, q: queue.Queue):
        """Stream tokens into a queue. Runs in a background thread."""
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},  # get authoritative token count at end
        }
        t_start = time.perf_counter()
        t_first = None
        server_tokens = None  # from usage field — correct for speculative decoding
        text = ""
        try:
            with req.post(f"{url}/v1/chat/completions", json=payload,
                          stream=True, timeout=120) as r:
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        # Capture server-reported token count (final chunk only)
                        usage = chunk.get("usage")
                        if usage and "completion_tokens" in usage:
                            server_tokens = usage["completion_tokens"]
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                            text += content
                            q.put(("token", content, now - t_start))
                    except Exception:
                        continue
        except Exception as e:
            q.put(("error", str(e), 0, 0))
            return

        total_time = time.perf_counter() - t_start
        # Use server-reported token count so TPOT is per-actual-token, not per-SSE-chunk
        tokens = server_tokens if server_tokens else max(len(text) // 4, 1)
        tpot = ((time.perf_counter() - t_first) / (tokens - 1) * 1000) if t_first and tokens > 1 else None
        ttft = (t_first - t_start) * 1000 if t_first else None
        q.put(("done", text, tokens, total_time, ttft, tpot))

    # ── BUTTON 1: Metrics benchmark ────────────────────────────────────────────
    col_btn1, col_btn2 = st.columns([1, 1])
    run_metrics  = col_btn1.button("📊  Run Metrics Benchmark", type="primary",
                                   use_container_width=True)
    run_race     = col_btn2.button("🏁  Live Token Race  (watch tokens stream)",
                                   use_container_width=True)

    # ── Metrics view ──────────────────────────────────────────────────────────
    if run_metrics:
        with st.spinner(f"Firing {concurrency} concurrent request(s) to both servers…"):
            b_results, e_results = asyncio.run(run_benchmark_async(prompt, concurrency))
            acceptance_rate = get_acceptance_rate(eagle3_url)

        b_tpot = avg(b_results, "tpot_ms")
        e_tpot = avg(e_results, "tpot_ms")
        b_tps  = avg(b_results, "tokens_per_sec")
        e_tps  = avg(e_results, "tokens_per_sec")
        b_ttft = avg(b_results, "ttft_ms")
        e_ttft = avg(e_results, "ttft_ms")
        b_cost = sum(r.get("cost_usd") or 0 for r in b_results)
        e_cost = sum(r.get("cost_usd") or 0 for r in e_results)

        if b_tpot and e_tpot:
            tpot_speedup = b_tpot / e_tpot
            tps_speedup  = (e_tps / b_tps) if (b_tps and e_tps) else None
            cost_savings = (1 - e_cost / b_cost) * 100 if b_cost else None

            if tpot_speedup >= 1.0:
                headline = f"⚡ EAGLE-3 is **{tpot_speedup:.2f}× faster** per token (TPOT)"
                if tps_speedup:
                    headline += f"  |  **{tps_speedup:.2f}× overall throughput**"
                if cost_savings and cost_savings > 0:
                    headline += f"  |  **{cost_savings:.0f}% cheaper** per run"
                if acceptance_rate:
                    headline += f"  |  Acceptance rate: **{acceptance_rate:.0%}**"
                st.success(headline)
            else:
                st.warning(
                    f"At concurrency={concurrency}, overhead outweighs gains "
                    f"(TPOT ratio: {tpot_speedup:.2f}×). Try concurrency=1."
                )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔵 Baseline (greedy decoding)")
            st.metric("TPOT — time per output token", f"{b_tpot:.1f} ms" if b_tpot else "N/A")
            st.metric("Tokens / sec",                 f"{b_tps:.1f}"     if b_tps  else "N/A")
            st.metric("TTFT — time to first token",   f"{b_ttft:.0f} ms" if b_ttft else "N/A")
            st.metric("GPU cost (this run)",           f"${b_cost:.5f}")
            st.metric("Acceptance rate",               "N/A — no draft model")
            if b_results[0].get("text"):
                st.text_area("Sample output", b_results[0]["text"][:500], height=150, key="b_text")

        with col2:
            st.subheader("🔴 EAGLE-3 (speculative decoding, k=5)")
            # delta: negative = improvement for TPOT/TTFT, positive = improvement for TPS
            tpot_delta = f"{e_tpot - b_tpot:+.1f} ms" if (b_tpot and e_tpot) else None
            tps_delta  = f"{e_tps - b_tps:+.1f} tok/s" if (b_tps and e_tps) else None
            ttft_delta = f"{e_ttft - b_ttft:+.0f} ms"  if (b_ttft and e_ttft) else None

            st.metric("TPOT — time per output token",
                      f"{e_tpot:.1f} ms" if e_tpot else "N/A",
                      delta=tpot_delta, delta_color="inverse")   # negative = green ✓
            st.metric("Tokens / sec",
                      f"{e_tps:.1f}" if e_tps else "N/A",
                      delta=tps_delta)                           # positive = green ✓
            st.metric("TTFT — time to first token",
                      f"{e_ttft:.0f} ms" if e_ttft else "N/A",
                      delta=ttft_delta, delta_color="inverse")
            st.metric("GPU cost (this run)", f"${e_cost:.5f}")
            st.metric("Draft acceptance rate",
                      f"{acceptance_rate:.1%}" if acceptance_rate else "fetching…")
            if e_results[0].get("text"):
                st.text_area("Sample output", e_results[0]["text"][:500], height=150, key="e_text")

        errors = [r for r in b_results + e_results if r.get("error")]
        if errors:
            st.error(f"{len(errors)} request(s) failed: {errors[0]['error']}")

        with st.expander("How to read these results"):
            st.markdown(
                "**TPOT is the key metric.** It measures how fast tokens stream after the first one. "
                "Lower TPOT = faster perceived generation for the user.\n\n"
                "**TTFT is often slightly higher for EAGLE-3** because the draft model needs an "
                "additional forward pass before the first accepted token is emitted. "
                "This tradeoff is almost always worth it at low-to-moderate concurrency.\n\n"
                "**Speedup shrinks at high concurrency** because the GPU becomes compute-bound "
                "and the draft verification overhead no longer amortizes."
            )

    # ── BUTTON 2: Live Token Race ──────────────────────────────────────────────
    if run_race:
        st.markdown("---")
        st.subheader("🏁 Token Race — watching generation in real time")
        st.caption("Both servers receive the same request at the same moment. "
                   "Watch EAGLE-3 pull ahead.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔵 Baseline (greedy)**")
            b_counter   = st.empty()
            b_output    = st.empty()
            b_stats_ph  = st.empty()
        with col2:
            st.markdown("**🔴 EAGLE-3 (speculative, k=5)**")
            e_counter   = st.empty()
            e_output    = st.empty()
            e_stats_ph  = st.empty()

        # Fire both in background threads
        b_q, e_q = queue.Queue(), queue.Queue()
        threading.Thread(target=stream_sync,
                         args=(baseline_url, "baseline", prompt, b_q), daemon=True).start()
        threading.Thread(target=stream_sync,
                         args=(eagle3_url,   "eagle3",  prompt, e_q), daemon=True).start()

        b_text, e_text = "", ""
        b_tokens, e_tokens = 0, 0
        b_done, e_done = False, False
        b_elapsed, e_elapsed = 0.0, 0.0
        b_tpot_live, e_tpot_live = None, None
        b_ttft_live, e_ttft_live = None, None

        winner_announced = False

        while not (b_done and e_done):
            # Drain baseline queue
            for _ in range(50):
                if b_done:
                    break
                try:
                    ev = b_q.get_nowait()
                    if ev[0] == "token":
                        _, content, b_elapsed = ev
                        b_text += content
                    elif ev[0] == "done":
                        _, _, b_tokens, b_elapsed, b_ttft_live, b_tpot_live = ev
                        b_done = True
                    elif ev[0] == "error":
                        b_done = True
                except queue.Empty:
                    break

            # Drain eagle3 queue
            for _ in range(50):
                if e_done:
                    break
                try:
                    ev = e_q.get_nowait()
                    if ev[0] == "token":
                        _, content, e_elapsed = ev
                        e_text += content
                    elif ev[0] == "done":
                        _, _, e_tokens, e_elapsed, e_ttft_live, e_tpot_live = ev
                        e_done = True
                    elif ev[0] == "error":
                        e_done = True
                except queue.Empty:
                    pass

            # Estimate tokens from actual text (accurate for both, ignores SSE chunk size)
            b_est_tok = len(b_text) // 4
            e_est_tok = len(e_text) // 4

            # Update UI
            b_counter.metric("~Tokens generated",
                              b_est_tok,
                              delta=f"{b_est_tok / b_elapsed:.0f} tok/s" if b_elapsed > 0 else None)
            e_counter.metric("~Tokens generated",
                              e_est_tok,
                              delta=f"{e_est_tok / e_elapsed:.0f} tok/s" if e_elapsed > 0 else None)
            b_output.markdown(
                f"<div style='background:#f0f4ff;padding:12px;border-radius:8px;"
                f"font-size:14px;min-height:120px;white-space:pre-wrap'>{b_text}</div>",
                unsafe_allow_html=True,
            )
            e_output.markdown(
                f"<div style='background:#fff4f0;padding:12px;border-radius:8px;"
                f"font-size:14px;min-height:120px;white-space:pre-wrap'>{e_text}</div>",
                unsafe_allow_html=True,
            )

            if b_elapsed > 0:
                b_stats_ph.caption(f"⏱ {b_elapsed:.1f}s elapsed")
            if e_elapsed > 0:
                e_stats_ph.caption(f"⏱ {e_elapsed:.1f}s elapsed")

            time.sleep(0.08)   # ~12 FPS

        # Final result banner
        if b_elapsed > 0 and e_elapsed > 0:
            speedup = b_elapsed / e_elapsed
            if speedup > 1.0:
                st.success(f"🏆 EAGLE-3 finished in **{e_elapsed:.2f}s** vs baseline **{b_elapsed:.2f}s** — "
                           f"**{speedup:.2f}× faster** wall-clock time for this request.")
            else:
                st.info(f"Baseline: {b_elapsed:.2f}s | EAGLE-3: {e_elapsed:.2f}s")

        if b_tpot_live and e_tpot_live:
            fc1, fc2 = st.columns(2)
            fc1.metric("Baseline TPOT", f"{b_tpot_live:.1f} ms")
            fc1.metric("Baseline TTFT", f"{b_ttft_live:.0f} ms" if b_ttft_live else "N/A")
            fc2.metric("EAGLE-3 TPOT", f"{e_tpot_live:.1f} ms",
                       delta=f"{e_tpot_live - b_tpot_live:+.1f} ms", delta_color="inverse")
            fc2.metric("EAGLE-3 TTFT", f"{e_ttft_live:.0f} ms" if e_ttft_live else "N/A",
                       delta=f"{(e_ttft_live or 0) - (b_ttft_live or 0):+.0f} ms",
                       delta_color="inverse")

        ar = get_acceptance_rate(eagle3_url)
        if ar:
            st.info(f"Draft token acceptance rate: **{ar:.1%}** — "
                    f"theoretical speedup with k=5: **{(1 + ar*5)/(1+ar):.2f}×**")


# ── RESULTS ANALYSIS MODE ──────────────────────────────────────────────────────
else:
    st.title("Results Analysis — Full Sweep Data")

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    @st.cache_data
    def load_results():
        records = []
        for p in RESULTS_DIR.glob("*.jsonl"):
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r.get("ttft_ms") is not None:
                            records.append(r)
                    except Exception:
                        continue
        return pd.DataFrame(records)

    if not any(RESULTS_DIR.glob("*.jsonl")):
        st.warning("No results found in results/raw/. Run the benchmark sweep first.")
    else:
        df = load_results()
        st.success(f"Loaded {len(df)} request records from {RESULTS_DIR}")

        c1, c2, c3 = st.columns(3)
        with c1:
            gpu_filter = st.multiselect("GPU", df["gpu_type"].unique().tolist(),
                                        default=df["gpu_type"].unique().tolist())
        with c2:
            task_filter = st.multiselect("Task", df["task"].unique().tolist(),
                                         default=df["task"].unique().tolist())
        with c3:
            sys_filter = st.multiselect("System", df["system"].unique().tolist(),
                                        default=df["system"].unique().tolist())

        df_f = df[df["gpu_type"].isin(gpu_filter) &
                  df["task"].isin(task_filter) &
                  df["system"].isin(sys_filter)]

        agg = df_f.groupby(["system", "gpu_type", "task", "concurrency"])[
            ["ttft_ms", "tpot_ms", "tokens_per_sec", "gpu_cost_usd"]
        ].mean().reset_index()

        st.subheader("Time to First Token vs Concurrency")
        for gpu in agg["gpu_type"].unique():
            for task in agg["task"].unique():
                sub = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                for system, grp in sub.groupby("system"):
                    grp = grp.sort_values("concurrency")
                    ax.plot(grp["concurrency"], grp["ttft_ms"], marker="o",
                            label=system, linewidth=2)
                ax.set_title(f"TTFT — {task} on {gpu}")
                ax.set_xlabel("Concurrent Requests")
                ax.set_ylabel("TTFT (ms)")
                ax.set_xticks([1, 4, 8, 16, 32])
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Throughput vs Concurrency")
        for gpu in agg["gpu_type"].unique():
            for task in agg["task"].unique():
                sub = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                for system, grp in sub.groupby("system"):
                    grp = grp.sort_values("concurrency")
                    ax.plot(grp["concurrency"], grp["tokens_per_sec"], marker="s",
                            label=system, linewidth=2)
                ax.set_title(f"Tokens/sec — {task} on {gpu}")
                ax.set_xlabel("Concurrent Requests")
                ax.set_ylabel("Tokens / Second")
                ax.set_xticks([1, 4, 8, 16, 32])
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        with st.expander("Raw aggregated data"):
            st.dataframe(agg)

        st.subheader("Cost Analysis")
        df_c = df_f[df_f["output_tokens"] > 0].copy()
        df_c["cost_per_token"] = df_c["gpu_cost_usd"] / df_c["output_tokens"]
        ct = df_c.groupby(["system", "gpu_type"])["cost_per_token"].mean().reset_index()
        ct["cost_per_1k_tokens"] = ct["cost_per_token"] * 1000
        st.dataframe(ct[["system", "gpu_type", "cost_per_1k_tokens"]].round(6))
