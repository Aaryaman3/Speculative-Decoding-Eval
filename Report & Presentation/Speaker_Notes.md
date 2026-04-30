# Speaker Notes: Speculative Decoding Evaluation (Highly Detailed Edition)

**Total Target Duration:** ~20 Minutes
**Slide Count:** 12 Slides
**Pacing:** Approximately 1 minute 40 seconds per slide.
**Goal:** This version provides an extensive, deep-dive script with robust bullet points, exact metrics, and technical insights so you are fully armed for both the presentation and the Q&A.

---

## Slide 01: Title Slide & Introduction
**Estimated Time:** 1 minute 15 seconds

**Key Points to Hit:**
- Introduce all team members clearly.
- Define the exact project scope: Evaluating EAGLE-3.
- State the core research question: Does speculative decoding theory hold up under extreme real-world cloud concurrency?

**Expanded Speaker Script:**
> "Good morning/afternoon everyone. Today, our team—Aaryaman, Shreya, Raj, and myself, Himanshu—will present our final applied machine learning project: Characterizing EAGLE-3 Speculative Decoding in vLLM.
> 
> In academic literature, speculative decoding is highly praised as a 'free' way to speed up Large Language Models. However, most papers evaluate these algorithms in isolation, often for a single user. We set out to answer a much more practical, systems-level question: Does this theoretical speedup actually hold up in a multi-user, highly concurrent cloud serving environment? Furthermore, how does the underlying hardware architecture—from high-end Cloud GPUs down to unified memory on Apple Silicon—dictate the success or failure of this algorithm?"

---

## Slide 02: The Core Problem & Our Approach
**Estimated Time:** 1 minute 45 seconds

**Key Points to Hit:**
- Explain the "Memory-Bandwidth Bound" bottleneck (compute cores are idle waiting for weights).
- Explain standard speculative decoding (draft -> verify in parallel).
- Explain why EAGLE-3 is special (feature-level head vs separate autoregressive model).

**Expanded Speaker Script:**
> "To understand the necessity of our project, we first have to understand the core bottleneck in LLM inference. Standard autoregressive inference is fundamentally memory-bandwidth bound. The GPU spends the vast majority of its time moving model weights from VRAM to the compute cores, leaving those compute cores essentially idle. You are not bound by math; you are bound by memory speed.
> 
> Speculative decoding breaks this bottleneck. A smaller, faster 'draft model' guesses several tokens ahead. Then, the massive 8B target model takes those guessed tokens and verifies all of them in parallel during a single forward pass. Any tokens that are accepted give us a 'free' speedup without changing the mathematical distribution of the output.
> 
> We specifically chose the EAGLE-3 architecture over standard speculative draft models. Standard draft models (like a tiny 1B model) often guess poorly because their token distributions differ from the 8B target model. EAGLE-3 solves this by using a lightweight *feature-level draft head* attached directly to the target model itself. As we will see, this results in significantly higher token acceptance rates."

---

## Slide 03: Experimental Architecture
**Estimated Time:** 2 minutes

**Key Points to Hit:**
- The 45-configuration Matrix: 3 tasks × 3 hardware tiers × 5 concurrency levels.
- The 3 Tasks: ShareGPT (Chat), HumanEval (Code), CNN (Summarization).
- The Hardware Challenge: L4 OOM limits (20GB vs 22GB) forcing strict sequential orchestration.

**Expanded Speaker Script:**
> "We did not test this in a vacuum; we built a rigorous cross-platform evaluation matrix consisting of 45 total configurations.
> 
> We evaluated Llama-3.1-8B-Instruct across three extremely diverse tasks to see how output structure affects performance: Chat via ShareGPT, Code Generation via HumanEval, and Summarization via CNN/DailyMail. We swept through five concurrency levels, starting at a single user and scaling up to 32 simultaneous requests to saturate the servers. We also tested three distinct hardware tiers: the flagship A100 (80GB), the budget L4 (24GB), and Apple Silicon's unified memory.
> 
> Testing the budget L4 GPU presented a massive engineering constraint. The Llama 8B weights and KV cache consume about 20 gigabytes of VRAM. When you add the EAGLE-3 draft head, it jumps to 22 gigabytes. The L4 only has 24 gigabytes total. If we tried to run both servers simultaneously for load testing, the GPU would instantly crash with Out-Of-Memory errors. To solve this, we engineered a strict orchestration script to run a sweep, securely kill the server, flush the VRAM entirely, and boot the next configuration sequentially."

---

## Slide 04: The "Gotcha" — Debugging vLLM
**Estimated Time:** 2.5 minutes

**Key Points to Hit:**
- The Scare: Baseline 100 TPS vs EAGLE-3 72 TPS.
- The Investigation: Diving into raw JSONL logs to find `chars/chunk` discrepancies (7.7 vs 3.8).
- The Root Cause: vLLM batches multiple speculative tokens into one Server-Sent Event (SSE) chunk.
- The Fix: Using `stream_options: {include_usage: true}` to get authoritative server data.

**Expanded Speaker Script:**
> "Before diving into the final results, I want to share a critical engineering hurdle that almost derailed the project. After our first massive sweep, our data showed that EAGLE-3 was actually *slower* than the baseline. For example, Baseline was showing 100 tokens-per-second, while EAGLE-3 was reporting 72 tokens-per-second.
> 
> We refused to blindly accept this data. We dug directly into the raw JSON streaming logs. We calculated a metric we called `chars/chunk`. For the baseline, it was 7.7 characters per chunk. But for EAGLE-3, it was 3.8. 
> 
> We realized our load tester was mistakenly counting Server-Sent-Event (SSE) chunks as tokens. For standard decoding, 1 chunk equals 1 token. But vLLM is highly optimized—during speculative decoding, if it accepts 3 tokens instantly, it batches all 3 tokens into a *single* SSE chunk! By counting chunks on the client side, we were undercounting EAGLE-3's generation speed by nearly a factor of 2.
> 
> We patched our load tester to inject the `stream_options` payload flag to parse authoritative server-side usage metrics. Our data integrity was immediately restored, and the true power of EAGLE-3 emerged."

---

## Slide 05: Throughput & Speedup
**Estimated Time:** 1.5 minutes

**Key Points to Hit:**
- Direct attention to the Heatmap. Green/Blue = EAGLE-3 wins.
- Highlight the peak: 2.91x speedup on L4 (Code at c=1).
- Highlight the minimum: 1.20x speedup on L4 (Chat at c=1).
- Concurrency scaling: No crossover point was found up to c=32.

**Expanded Speaker Script:**
> "With the client-side bug fixed, the results were staggering. As you can see in this heatmap, where any value over 1.0 means EAGLE-3 is faster, EAGLE-3 is an unconditional win across the board on both the A100 and the L4.
> 
> We achieved massive speedups ranging from 2x to nearly 3x. Our absolute peak gain was in Code Generation on the L4 at concurrency 1, hitting a 2.91x speedup. Even our absolute worst-case scenario—open-ended Chat on the L4 at concurrency 1—still yielded a 1.2x speedup.
> 
> A major debate in academic literature is the 'crossover point'—the exact concurrency level where the overhead of the draft model destroys the performance gains. Even when pushing the servers to 32 concurrent requests and fully saturating the GPUs, we observed zero crossover points. EAGLE-3 remained vastly superior in every single configuration."

---

## Slide 06: Throughput Deep Dive
**Estimated Time:** 1 minute

**Key Points to Hit:**
- Direct visual evidence of the speedup via line graphs.
- Emphasize the sustained gap across all concurrency levels.

**Expanded Speaker Script:**
> "If we look directly at the raw Tokens Per Second on these line charts, the gap between the baseline and EAGLE-3 is undeniable.
> 
> You can visually trace how much higher the EAGLE-3 lines sit above the baseline across all three tasks. The generation speed difference isn't a small margin of error; it's a completely different class of performance. This proves that the theoretical mathematical benefits of speculative decoding map directly to massive real-world throughput gains under heavy server load."

---

## Slide 07: Task Variability & Acceptance Rates
**Estimated Time:** 2 minutes

**Key Points to Hit:**
- Acceptance rate dictates speedup.
- Hardware Agnostic: A100 and L4 give identical acceptance rates for the same prompt.
- Code (High structure): Highly predictable, 50% acceptance.
- Chat (Low structure): Starts at 19%, but climbs to 42% at c=32 due to batching patterns.

**Expanded Speaker Script:**
> "We've established that Code generation performs much better than Chat, but *why*? It all comes down to the Acceptance Rate—how often the large model agrees with the draft model's guesses.
> 
> Our first major finding here is that acceptance rates are strictly hardware-agnostic. The A100 and the L4 produced nearly identical acceptance rates, proving this is a pure property of the model pair and the prompt syntax, completely independent of compute power.
> 
> Code Generation has highly strict, predictable syntax (like function definitions and loops). The draft head guesses correctly about 50% of the time, driving the massive speedups. 
> 
> Conversely, conversational Chat is creative and unpredictable. For a single user, the acceptance rate is a terrible 19%. However, notice the trend: it climbs dramatically to 42% at high concurrency. We hypothesize that under heavy load, continuous batching groups similar sequences together, allowing the draft head to suddenly find structural patterns even in open-ended chat."

---

## Slide 08: TTFT Trade-off
**Estimated Time:** 1.5 minutes

**Key Points to Hit:**
- Time-To-First-Token (TTFT) vs Time-Per-Output-Token (TPOT).
- The c=1 minimal overhead (8ms).
- The c=4 'Bubble': TTFT spikes to 914ms (19x overhead) for Chat.
- Recovery at c=8.

**Expanded Speaker Script:**
> "There is no free lunch in computer science. The extra compute required for the draft model introduces a Time-To-First-Token (TTFT) overhead. 
> 
> At concurrency 1, this overhead is minimal. For Code generation, it only adds an 8-millisecond delay before the first token appears. 
> 
> However, at concurrency 4 for Chat, we discovered a strange 'scheduling bubble'. The TTFT spiked to over 900 milliseconds—a 19x overhead compared to the baseline. The system was struggling to batch the initial draft passes efficiently. But interestingly, by concurrency 8, this bubble vanishes entirely, and the baseline and EAGLE-3 TTFT normalize again.
> 
> The ultimate takeaway is this: while the initial TTFT might see a slight scheduling bump, the Time Per Output Token (the generation speed) always improves drastically once the stream begins."

---

## Slide 09: Cost-Performance Frontier
**Estimated Time:** 1.5 minutes

**Key Points to Hit:**
- 2x faster = 50% less GPU compute time per request.
- A100 savings: 40.7% cheaper.
- L4 savings: 50.6% cheaper.
- The Sweet Spot: L4 + EAGLE matches A100 Baseline throughput at 66% lower cost.

**Expanded Speaker Script:**
> "In cloud computing, throughput is not just a technical metric; it translates directly to economics. 
> 
> Because EAGLE-3 pushes tokens out 2 to 3 times faster, the GPU spends significantly less time computing each request. This translates to an immediate 40 to 50 percent reduction in cloud compute costs per request. 
> 
> We mapped out a Cost-Performance Frontier and found the ultimate 'sweet spot' for budget-conscious deployments: The L4 GPU with EAGLE-3 enabled. It effectively matches the baseline throughput of the flagship A100 GPU—about 90 tokens per second—but costs 66% less per request. It's the ultimate cloud engineering budget hack."

---

## Slide 10: The Apple Silicon Reality
**Estimated Time:** 1.5 minutes

**Key Points to Hit:**
- Success at c=1: 1.19x to 1.53x speedup.
- Failure at c>=4: Severe regressions.
- The Unfairness Problem: CV=188% at c=32. 150ms vs 45,000ms waits.

**Expanded Speaker Script:**
> "We also evaluated Apple Silicon's unified memory architecture using the MLX framework, and the story here is entirely different.
> 
> If you are a single user running a local agent on your Mac, it works beautifully—yielding a genuine 1.2x to 1.5x speedup. 
> 
> But the moment you introduce concurrency, it becomes a multi-user catastrophe. At a concurrency of just 4, throughput drops below the baseline across all tasks. By concurrency 32, we encountered an extreme unfairness problem. The Coefficient of Variation hit 188%. The fastest 10% of users received their first token in 150 milliseconds, while the slowest 10% in the exact same batch waited an agonizing 45 seconds."

---

## Slide 11: The Architecture Insight
**Estimated Time:** 2.5 minutes

**Key Points to Hit:**
- Why did Mac fail? The Scheduler.
- vLLM = Continuous Batching (Parallel draft/verify).
- MLX = Serial Passes (Queue explosion).
- The failure is architectural, not algorithmic.

**Expanded Speaker Script:**
> "Why did the exact same theoretical algorithm succeed brilliantly on the Cloud GPUs but fail completely on the Mac? This is the biggest architectural insight of our research.
> 
> The difference has nothing to do with the models—it is entirely dictated by the serving framework's scheduler. 
> 
> vLLM on the Nvidia GPUs uses Continuous Batching. It intelligently batches the draft passes and the verification passes across all concurrent requests simultaneously, ensuring the GPU stays perfectly saturated.
> 
> The MLX framework on the Mac, however, processes speculative passes serially. Each individual request must run its entire draft-and-verify cycle before the next request can begin. When multiple users hit the Mac simultaneously, there is zero cross-request batching, and the queue depth explodes. 
> 
> The regressions we saw on the Mac are an architectural scheduling failure, not a failure of speculative decoding theory."

---

## Slide 12: Conclusions & Recommendations
**Estimated Time:** 1.5 minutes

**Key Points to Hit:**
- 1. Cloud GPUs: Unconditional win, enable by default.
- 2. Code Generation: Peak winner.
- 3. Apple Silicon: Single-user edge deployment only.
- Final wrap-up.

**Expanded Speaker Script:**
> "To summarize our actionable findings:
> 
> First, on Cloud GPUs like the A100 and L4 running vLLM, EAGLE-3 is an unconditional win. There are zero crossover regressions up to 32 users. Turn it on, and leave it on in production.
> 
> Second, if you are serving Code Generation, you win the biggest. Highly structured output maximizes the acceptance rate, yielding nearly a 3x speedup and cutting your costs in half.
> 
> Finally, for Edge deployments on Apple Silicon using MLX, speculative decoding must be strictly limited to single-user local agents until the framework's scheduler is rewritten to support continuous batching.
> 
> Thank you so much for your time today. We'd love to open the floor to any questions."
