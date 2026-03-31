# Mechanistic Interpretability: First-Principles Curriculum

> **Philosophy:** Learn the minimum viable basics, then immediately do research. Mech interp is an empirical science — you learn by doing, not by reading.
>
> **Time Estimate:** 10–12 weeks full-time (~40 hrs/week), or ~20 weeks part-time (~20 hrs/week)
>
> **Prerequisites:** Comfortable with Python. Familiar with basic calculus, probability, and introductory ML concepts (what a neural net is, what training means).

---

## How This Curriculum Works

This curriculum is structured in three phases, directly mirroring Neel Nanda's recommended progression:

| Phase | Duration | Goal |
|-------|----------|------|
| **Phase 1: Build the Machine** | Weeks 1–4 | Code transformers and core interp tools from scratch. Gain fluency in linear algebra, PyTorch, and the mech interp stack. |
| **Phase 2: Mini-Projects** | Weeks 5–7 | Practice research with throwaway 2–5 day projects. Focus on fast-feedback skills: running experiments, interpreting results, getting unstuck. |
| **Phase 3: Full Research Sprints** | Weeks 8–12 | Work in 1–2 week sprints. Practice ideation, deeper understanding, skepticism, and writing up results. |

**Rules:**

1. **No passive reading.** Every paper assigned has a concrete coding or analytical task attached. If a paper appears below, it is because you need it to build something or bridge a specific gap.
2. **Build before you import.** Implement core algorithms (attention, backprop, probes, SAE forward pass) from raw tensors before using library abstractions.
3. **LLMs are tutors, not crutches.** Use frontier LLMs (Gemini 2.5 Pro, Claude Opus, GPT-5 Thinking) extensively for explanations, feedback, and exercises — but never copy-paste code during learning exercises. Use anti-sycophancy prompts for feedback.
4. **Post-mortems are mandatory.** After every project (even a 2-day one), spend ≥30 minutes writing what you tried, what worked, what you'd do differently.

---

## Phase 1: Build the Machine (Weeks 1–4)

The goal is not to finish learning. It is to learn enough that all future learning happens through research.

---

### Module 1 — Linear Algebra & Transformer Internals (Week 1)

**Time:** ~40 hours

**Why this matters:** Mech interp is applied linear algebra. If you can't think in vectors and matrices fluently — what a basis change means, what SVD reveals, why low-rank matrices behave differently — you will hit a wall on every single project. This is the single highest-ROI math investment.

#### 1A: Linear Algebra Foundations (~15 hrs)

**Do:**
- [X] Watch 3Blue1Brown's *Essence of Linear Algebra* (full series). Take notes on geometric intuitions, not formulas.
- [X] After each video, summarize the key insight in your own words to an LLM and ask for critical feedback using an anti-sycophancy prompt.

**Prove you understand — complete all of the following:**

1. **Written explanations** (submit to an LLM for harsh critique):
   - [X] What does it mean geometrically to multiply a vector by a matrix? What is a basis change and why does it matter for interpretability?
   - [X] Explain SVD: what the three matrices represent, why it reveals the "true structure" of a linear map, and how rank relates to the number of non-zero singular values.
   - [X] Explain eigenvalues/eigenvectors. What makes an orthonormal basis special?
   - [ ] How does a low-rank matrix differ from a full-rank matrix? Why does this matter when analyzing weight matrices inside transformers?

2. **Coding exercises** (np only, no library SVD/eigendecomp — implement the power iteration method yourself for eigenvalues):
   - [X] Implement matrix-vector multiplication from scratch
   - [X] Implement a change-of-basis operation
   - [ ] Compute rank via row reduction
   - [ ] Implement power iteration to find the dominant eigenvector of a matrix

> **LLM exercise:** Put the *Mathematical Framework for Transformer Circuits* (Elhage et al., 2021 — transformer-circuits.pub) in the context window. Ask the LLM to generate 10 exercises testing your intuitions about how linear algebra appears inside transformers (residual stream as a vector space, attention as low-rank operations, etc.). Do all 10.
> - [ ] Complete all 10 LLM-generated exercises

#### 1B: Build a Transformer from Scratch (~25 hrs)

**Do:**
- [ ] Watch Neel Nanda's two video tutorials on implementing GPT-2 from scratch (start from basics).
- [ ] Complete **ARENA Chapter 1.1** — code a GPT-2-scale transformer in PyTorch. Do NOT copy-paste. Type every line.

**Required output — a single Python file containing:**

- [ ] Token + positional embeddings
- [ ] Multi-head self-attention (implement scaled dot-product attention from raw matrix multiplications, not `nn.MultiheadAttention`)
- [ ] Layer normalization
- [ ] MLP blocks (GELU activation)
- [ ] Residual connections
- [ ] Unembedding / logit computation
- [ ] Load pretrained GPT-2 Small weights into your architecture (from HuggingFace)
- [ ] Generate coherent text from a prompt
- [ ] **Annotate every component** with a comment explaining its role in the residual stream view of transformers

**Checkpoint:**
- [ ] You can explain, from memory, the complete forward pass of a transformer: what each component does to the residual stream, what the shapes of all tensors are at each step, and why attention is described as "information movement" while MLPs are "information processing."

**Stretch (optional but recommended):**
- [ ] Draw the full architecture diagram from memory, including tensor shapes at each stage.

---

### Module 2 — Core Mech Interp Techniques (Week 2)

**Time:** ~40 hours

**Why this matters:** Most mech interp research involves knowing which technique to apply and when. This week you build the core toolkit by implementing each technique yourself.

#### 2A: TransformerLens + Direct Observation (~10 hrs)

**Do:**
- [ ] Complete **ARENA Chapter 1.2, Sections 1–3** (TransformerLens basics, direct observation, hooking activations).
- [ ] Set up your GPU environment (Google Colab to start, then migrate to RunPod/Vast.ai).

**Required output:**
- [ ] A notebook that loads GPT-2 Small via TransformerLens and extracts residual stream activations at every layer for a given prompt
- [ ] Visualizes attention patterns for all heads at a chosen layer
- [ ] Computes the direct logit attribution (DLA) of each attention head and MLP layer on a specific token prediction
- [ ] Implements logit lens: project the residual stream at each layer through the unembedding matrix and show the top-5 predicted tokens evolving across layers

#### 2B: Activation Patching (~12 hrs)

**Do:**
- [ ] Complete **ARENA Chapter 1.4.1** (Activation Patching).
- [ ] Read the abstract + methods section only of *Towards Best Practices of Activation Patching* (Kramár et al., 2024) — skim for which metrics and corruption methods the field has converged on.

**Required output:**
- [ ] Implement activation patching from scratch on GPT-2 Small:
  - [ ] Choose a simple task (e.g., "The Eiffel Tower is in" → "Paris")
  - [ ] Create a clean and corrupted prompt pair
  - [ ] Patch activations from clean → corrupted run at each (layer, position) pair
  - [ ] Plot a heatmap showing which (layer, position) pairs are causally important for the correct answer
- [ ] Implement **attribution patching** (the linearized approximation) and compare the heatmap to your full patching results. Are they similar? Where do they diverge?

#### 2C: Linear Probes (~8 hrs)

**Do:**
- [ ] Implement a linear probe from scratch (a single `nn.Linear` layer trained with SGD on frozen model activations).

**Required output:**
- [ ] Create a small dataset of 200 true/false factual statements (use an LLM API to generate these)
- [ ] Extract residual stream activations at the last token position from each layer
- [ ] Train a logistic regression probe at each layer
- [ ] Plot accuracy by layer — at which layer does the model "know" truth?

**Paper checkpoint (targeted read):**
- [ ] Skim *The Geometry of Truth* (Marks et al., 2023) — focus on Section 3 (methods) and Section 4 (results). Your probe implementation should mirror their approach. Note the key finding: truth has a linear representation.

#### 2D: Steering Vectors (~10 hrs)

**Do:**
- [ ] Implement the full steering vector pipeline:

**Required output:**
- [ ] Use an LLM API (OpenRouter) to generate 32 happy and 32 sad prompts
- [ ] Run all prompts through GPT-2 Small, collect residual stream activations at the middle layer
- [ ] Compute the mean difference vector (happy − sad)
- [ ] Add this vector (scaled by various coefficients: 0, 1, 3, 5, 10) to the residual stream during generation
- [ ] Use an LLM API to rate the happiness of outputs at each coefficient
- [ ] Plot coefficient vs. happiness score
- [ ] **Critical exercise:** Try using a random vector of the same norm instead. Does it steer? (It shouldn't. If it does, your pipeline has a bug.)

---

### Module 3 — SAEs, Tooling & the Landscape (Week 3)

**Time:** ~40 hours

#### 3A: Sparse Autoencoders — Understand and Use (~20 hrs)

**Why SAEs matter:** SAEs decompose model activations into interpretable features, working around the polysemanticity problem. You need to understand what they do, their strengths and severe limitations, and how to use pre-trained SAEs. You do NOT need to train one from scratch right now.

**Do:**
- [ ] Complete **ARENA Chapter 1.3.2** (SAEs) — skip Section 1 (training details), focus on Sections 2–4 (using SAEs, Neuronpedia, interpreting features).
- [ ] Read the abstract + Section 2 (method) of *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (Cunningham et al., 2023). Understand the architecture: encoder, decoder, sparsity penalty.

**Required output:**
- [ ] **Implement the SAE forward pass from scratch** (encoder: ReLU(W_enc @ x + b_enc), decoder: W_dec @ z + b_dec, loss: reconstruction + L1 sparsity):
  - [ ] Don't train it — verify your architecture matches the paper by loading pre-trained SAE weights from SAELens and checking that your forward pass reproduces their outputs
- [ ] **Use Neuronpedia** to explore features of Gemma 2 2B:
  - [ ] Pick 5 features. For each, write a one-sentence hypothesis of what concept the feature represents
  - [ ] Find the max-activating dataset examples. Do they confirm your hypothesis?
  - [ ] Find at least one feature that seems polysemantic (activates on multiple unrelated concepts) — document it
- [ ] **SAE-based investigation:** Pick a prompt where GPT-2 Small or Gemma 2B makes a surprising prediction. Use an SAE to identify which features are most active. Can you explain the prediction?

#### 3B: Attribution Graphs & Neuronpedia (~8 hrs)

**Do:**
- [ ] Explore attribution graphs on Neuronpedia (available for Gemma 2B).

**Required output:**
- [ ] Pick 3 different prompts on Neuronpedia's attribution graph tool. For each:
  - [ ] Identify the top 3 features (nodes) driving the final prediction
  - [ ] Trace back: what earlier features feed into them?
  - [ ] Write a one-paragraph hypothesis of the model's "reasoning" for each prompt
- [ ] **Validation exercise:** Take one of your hypotheses and test it with black-box methods (change the prompt to see if behavior changes as predicted).

#### 3C: LLM APIs & Black-Box Methods (~7 hrs)

**Do:**
- [ ] Set up an OpenRouter account and write a reusable Python function for calling LLM APIs programmatically.
- [ ] Practice token forcing / prefill attacks (putting words in the model's mouth).

**Required output:**
- [ ] A utility module (`llm_utils.py`) with functions for:
  - [ ] Sending a prompt to any model via OpenRouter and getting a response
  - [ ] Batch-scoring: given N texts and a criterion, have an LLM rate each 1–10
  - [ ] Generating synthetic datasets: given a schema and examples, produce N new examples
- [ ] **Black-box investigation:** Pick a topic where a chat model (e.g., Qwen3) behaves interestingly:
  - [ ] Systematically vary the prompt to isolate what triggers the behavior
  - [ ] Use token forcing to test if the model "wants" to say something different
  - [ ] Write up your findings in 1 page

#### 3D: Literature Orientation (~5 hrs)

**Do:**
- [ ] Skim Neel Nanda's [list of favourite papers](https://www.neelnanda.io/mechanistic-interpretability/favourite-papers). Read summaries, not full papers. Use an LLM to explain any you find interesting.
- [ ] Skim the abstract and intro of *Open Problems in Mechanistic Interpretability* (the multi-author review). Put it in an LLM context window and ask: "What are the 5 most important open problems, and which have seen the most progress?"

**Required output:**
- [ ] A personal "landscape document" — a 2-page summary in your own words of:
  - [ ] The 5 mech interp techniques you've learned so far, when to use each, and their limitations
  - [ ] The 3 research directions you find most exciting and why
  - [ ] 2 papers you want to deep-dive on later

---

### Module 4 — Integration & Research Readiness (Week 4)

**Time:** ~40 hours

#### 4A: Deep-Dive Paper Study (~12 hrs)

**Do:**
- [ ] Choose ONE paper for a full deep dive:
  - [ ] *Interpretability in the Wild: IOI in GPT-2 Small* (Wang et al., 2022) — if you want to understand circuit analysis
  - [ ] *A Mechanistic Understanding of Alignment Algorithms: DPO and Toxicity* (Lee et al., 2024) — if you want safety-relevant interp
  - [ ] *Emergent Misalignment* (Betley et al., 2025) — if you want to study weird model behaviors

**Required output:**
- [ ] Read the full paper (with LLM assistance for context/questions as you go).
- [ ] Write a structured summary (1–2 pages):
  - [ ] What problem does this paper solve? Why should anyone care?
  - [ ] What is the core method? Could you reimplement the key experiment?
  - [ ] What are the limitations the authors acknowledge? What limitations do they NOT acknowledge?
  - [ ] What would you do differently or extend?
- [ ] **Replication attempt:** Reproduce the paper's most important figure or result.

#### 4B: Environment & Workflow Setup (~8 hrs)

**Do:**
- [ ] Set up Cursor (with $20/month plan if possible) for AI-assisted coding
- [ ] Set up cloud GPU (RunPod or Vast.ai) — verify you can load and run Qwen3-4B
- [ ] Install and test TransformerLens and nnsight
- [ ] Set up a research log template (Google Doc, Notion, or a Discord channel to yourself)

**Required output:**
- [ ] A "hello world" notebook on your cloud GPU that:
  - [ ] Loads Qwen3-4B (or Gemma 3 4B) via TransformerLens or nnsight
  - [ ] Extracts activations at a given layer
  - [ ] Applies logit lens
  - [ ] Uses your LLM API utility to score something about the model's output
- [ ] A research log with your first entry: reflections on Phase 1, what you found hardest, what you're most curious about

#### 4C: Capstone Exercise — End-to-End Mini-Investigation (~20 hrs)

**Do:**
- [ ] Pick one investigation (or invent your own):
  - [ ] **Refusal direction replication:** Use activation patching + steering vectors to find and manipulate the "refusal direction" in a model that refuses certain prompts.
  - [ ] **Induction head hunt on a new model:** Use TransformerLens to find induction heads in Qwen3-4B or Gemma 3.
  - [ ] **Probe a specific concept:** Train truth/sentiment/toxicity probes across all layers of a model.

**Required output:**
- [ ] A research log with entries for each day
- [ ] At least 5 plots/visualizations
- [ ] A 1-page summary of your findings (even if they're null results)

**Post-mortem (mandatory):**
- [ ] What worked? What was the biggest waste of time? What would you do differently? Where did you get stuck, and how did you get unstuck?

---

## Phase 2: Mini-Projects (Weeks 5–7)

You now have the toolkit. The goal is to do 3–5 throwaway projects of 2–5 days each. You are NOT trying to produce publishable work. You are trying to learn the craft of research: exploring efficiently, testing hypotheses skeptically, and getting unstuck.

**Rules for Phase 2:**
- Maximum 5 days per project. Then move on (unless something extraordinary is happening).
- After each project, do a written post-mortem (≥30 min).
- Don't stress about having the "best" idea. Just pick one and go.
- Practice both exploration-heavy projects (play with something, figure out what's going on) and understanding-heavy projects (test a specific hypothesis).

---

### Project Menu (pick 3–5 across Weeks 5–7)

#### Project A: Extend the Geometry of Truth (~3 days)
**Skills:** Probing, dataset creation, evaluation

- [ ] Replicate truth probes from *Geometry of Truth* on a modern model (Qwen3-4B or Gemma 3)
- [ ] Test whether probes generalize to questions in a different domain than they were trained on
- [ ] Test with ambiguous or contested statements
- [ ] Find a "truth direction" using PCA and try to steer the model toward lying
- [ ] Post-mortem (≥30 min)

#### Project B: Attribution Graphs Deep Dive (~3 days)
**Skills:** Attribution graphs, scientific mindset, hypothesis testing

- [ ] Use Neuronpedia's attribution graphs on Gemma 2B to form 3 hypotheses about how the model processes a specific type of prompt
- [ ] Test each hypothesis using patching, probing, and prompt variation
- [ ] Post-mortem (≥30 min)

#### Project C: Emergent Misalignment Exploration (~4 days)
**Skills:** SAEs, steering vectors, fine-tuning analysis, open-ended exploration

- [ ] Skim *Emergent Misalignment* (Betley et al., 2025) — focus on Section 2 (setup) and Section 4 (analysis)
- [ ] Download the open-source emergent misalignment models
- [ ] Identify SAE features that distinguish the misaligned model from the base model
- [ ] Find a steering vector that "fixes" the misalignment (or makes it worse)
- [ ] Analyze the model's chain of thought — can prompt engineering overcome the misalignment?
- [ ] Post-mortem (≥30 min)

#### Project D: Reasoning Model Interpretability (~4 days)
**Skills:** Black-box methods, CoT analysis, working with modern models

- [ ] Skim *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful* (Arcuschin et al., 2025) — focus on experimental methodology
- [ ] Pick prompts where a reasoning model produces surprising chains of thought
- [ ] Resample the second half of the CoT from the same first half — does the conclusion change?
- [ ] Delete "Wait, let me reconsider" sentences and regenerate — does it still reconsider?
- [ ] Identify patterns in when the CoT is faithful vs. unfaithful to the model's actual computation
- [ ] Post-mortem (≥30 min)

#### Project E: Taboo Word Model Forensics (~3 days)
**Skills:** Logit lens, SAEs, black-box methods, breadth of techniques

- [ ] Apply logit lens to find the secret word
- [ ] Apply SAE feature inspection
- [ ] Apply activation patching
- [ ] Apply prompt engineering / token forcing
- [ ] Apply steering vectors
- [ ] Document which techniques worked and which didn't, and why
- [ ] Post-mortem (≥30 min)

#### Project F: Build a Simple Interpretability Agent (~5 days)
**Skills:** LLM APIs, automated interpretability, coding

- [ ] Build an agent that runs logit lens and identifies interesting layers
- [ ] Extract top SAE features at those layers
- [ ] Look up feature descriptions
- [ ] Use an LLM to synthesize a natural-language explanation of "what the model is thinking"
- [ ] Test on 10 diverse prompts and assess plausibility
- [ ] Post-mortem (≥30 min)

---

### Phase 2 Reflection (End of Week 7)

- [ ] **Skill audit:** Review the fast/medium/slow skill list from Neel's guide. Rate yourself 1–5 on each. Which improved most? Which are weakest?
- [ ] **Interest mapping:** Which project excited you most? What direction would you want to spend 2 weeks on?
- [ ] **Generate 20 research ideas** in a single brainstorm session (quantity over quality). Rate each 1–10 on gut feel. Keep the top 5.

---

## Phase 3: Research Sprints (Weeks 8–12)

You are now doing real research. Work in 1–2 week sprints. After each sprint, do a thorough post-mortem and make a deliberate decision: continue or pivot? The default should be to pivot unless the project has genuine momentum.

**New priorities in this phase:**
- **Ideation:** Practice generating and evaluating research ideas
- **Skepticism:** Every exciting result is false until proven otherwise
- **Literature awareness:** Use LLMs with web search to check for prior work before and during each project
- **Writing:** Aim to produce at least one public write-up (blog post or short paper) by the end

---

### Sprint 1 (Weeks 8–9): Guided Research Direction

Pick one of the following high-ROI directions:

#### Option 1: Auditing Game — Build and Test

- [ ] Read *Sleeper Agents* (Hubinger et al., 2024) — understand how deceptive models are created
- [ ] Skim *Simple Probes Can Catch Sleeper Agents* for the interpretability response
- [ ] Fine-tune a small model to have a hidden behavior
- [ ] Attempt to find the hidden behavior using only black-box methods
- [ ] Attempt using mech interp (patching, probes, SAEs)
- [ ] Compare: which approach finds it faster and more reliably?
- [ ] If possible, have a friend try the black-box approach blind

#### Option 2: Linear Probes for Safety Monitoring

- [ ] Choose a safety-relevant concept (toxicity, deception, refusal, harmful intent)
- [ ] Train probes across multiple layers of 2–3 different models
- [ ] Evaluate: precision, recall, false positive rate
- [ ] Compare against prompting-based classifiers (the baseline)
- [ ] Test whether probes generalize across prompt distributions; document where they fail

#### Option 3: Reasoning Model Internals

- [ ] Compare internal representations of a reasoning model vs. the same base model during identical tasks
- [ ] Investigate whether reasoning models use their residual stream differently
- [ ] Identify features active during "productive" vs. "unproductive" reasoning steps
- [ ] Examine internal representations at the boundary between "thinking" and "answering"

**Sprint 1 wrap-up:**
- [ ] Research log with entries for each day
- [ ] 5+ key figures
- [ ] 1-page summary of findings
- [ ] Post-mortem

---

### Sprint 2 (Weeks 10–11): Independent Direction

- [ ] Spend 2 hours using LLM-powered literature search to check if your idea has been done before
- [ ] For every exciting result: set a 5-minute timer and brainstorm all the ways it could be false. Then test at least 2 of them.
- [ ] Compare results against the simplest possible baseline
- [ ] Research log with entries for each day
- [ ] Key figures
- [ ] 1-page summary of findings
- [ ] Post-mortem

---

### Sprint 3 (Week 12): Write-Up & Distillation

- [ ] **Narrative formation (Day 1):** Identify your 1–3 key claims, the evidence supporting them, and why anyone should care. Write as a single paragraph.
- [ ] **Abstract / TL;DR (Day 1):** Compress the narrative into 5–8 sentences.
- [ ] **Bullet-point outline (Day 2):** Structure the full piece — introduction, methods, results, discussion, limitations. Get LLM feedback with an anti-sycophancy prompt.
- [ ] **Figures (Days 2–3):** Make all key figures. Each should be interpretable without reading the text.
- [ ] **Draft (Days 3–4):** Write the full prose.
- [ ] **Revision (Day 5):** Get LLM feedback. Get feedback from a human if possible. Revise.
- [ ] Publish: a blog post (on LessWrong, personal blog, or a Google Doc) OR an Arxiv-style paper (1,500–4,000 words)
- [ ] Share with at least one person for feedback before publishing

---

## Paper Reference Index

Papers are assigned ONLY where they serve a specific learning objective. This is the complete list.

| Paper | Where Used | Why |
|-------|-----------|-----|
| *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021) | Module 1A | LLM context window for linear algebra exercises on transformer internals |
| *Towards Best Practices of Activation Patching* (Kramár et al., 2024) | Module 2B | Learn which patching metrics/corruption methods the field uses |
| *The Geometry of Truth* (Marks et al., 2023) | Module 2C, Project A | Methods reference for truth probing; replication target |
| *Sparse Autoencoders Find Highly Interpretable Features* (Cunningham et al., 2023) | Module 3A | Understand SAE architecture to verify your from-scratch implementation |
| *Open Problems in Mechanistic Interpretability* (multi-author, 2024) | Module 3D | Landscape orientation — skim with LLM, don't read cover to cover |
| *Emergent Misalignment* (Betley et al., 2025) | Module 4A or Project C | Deep dive target; model organisms to explore |
| *A Mechanistic Understanding of Alignment: DPO and Toxicity* (Lee et al., 2024) | Module 4A (option) | Deep dive target showing safety-relevant mech interp |
| *IOI Circuit* (Wang et al., 2022) | Module 4A (option) | Deep dive target for circuit analysis methodology |
| *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful* (Arcuschin et al., 2025) | Project D | Experimental methodology for CoT faithfulness investigation |
| *Sleeper Agents* (Hubinger et al., 2024) | Sprint 1, Option 1 | Understand model organisms for deception; probe-catching follow-up |
| *Simple Probes Can Catch Sleeper Agents* (Anthropic, 2024) | Sprint 1, Option 1 | Interpretability applied to detecting planted deception |
| *Transcoders Find Interpretable LLM Feature Circuits* (Dunefsky et al., 2024) | Reference | Background for attribution graph understanding (read if doing attribution graph work) |

---

## Appendix A: Key Resources

**Video / Tutorials:**
- 3Blue1Brown — Essence of Linear Algebra (full series)
- Neel Nanda — Implementing GPT-2 from scratch (2 video tutorials)
- Neel Nanda — YouTube research walkthroughs (for exploration mindset)
- ARENA Chapters 1.1, 1.2 (Sections 1–3), 1.3.2, 1.4.1

**Tools:**
- TransformerLens (small models ≤9B, complex experiments)
- nnsight (large models, HuggingFace wrapper)
- SAELens + Neuronpedia (SAE features, attribution graphs)
- OpenRouter (LLM API access)
- Cursor (AI-assisted coding)
- RunPod / Vast.ai (cloud GPUs)

**Models to work with (as of early 2026):**
- GPT-2 Small (learning / tutorials)
- Gemma 2 2B or Gemma 3 4B (SAE ecosystem, Neuronpedia support)
- Qwen3 family (good default modern models, reasoning + non-reasoning modes)

**Context files for LLM queries:**
- Neel Nanda's saved context files for mech interp (linked in his guide)
- For general queries, use his default context file

**Community:**
- Open Source Mech Interp Slack
- Eleuther Discord (#interpretability-general)
- Mech Interp Discord
- LessWrong / AlignmentForum (for publishing and staying current)

---

## Appendix B: What NOT to Spend Time On

Neel Nanda explicitly warns against these fads. Avoid them unless you have a specific, novel angle:

1. **Interpreting toy models on algorithmic tasks** (e.g., grokking, modular addition) — insights don't generalize to real models. The field has moved on.
2. **Basic circuit analysis** (finding a sparse subgraph of attention heads/MLPs responsible for a task) — the technique is worth *learning*, but simply "finding a circuit" is no longer a novel contribution. You need to use the circuit to reveal something deeper.
3. **Incremental SAE improvements** (new architectures, new training tricks, "what if we use SAEs for X?") — the field is saturated. SAEs are a useful tool, not a research agenda.
4. **Reading dozens of papers before writing code** — mech interp is empirical. If you've been reading for more than 2 days without running an experiment, stop and build something.

---

## Appendix C: Weekly Time Budget Template

| Activity | Hours/Week |
|----------|-----------|
| Coding (implementation, experiments) | 20–25 |
| Reading (papers, docs, tutorials) | 5–8 |
| LLM-assisted learning (exercises, feedback) | 4–6 |
| Reflection (post-mortems, research log) | 2–3 |
| Environment / tooling setup | 1–2 |
| **Total** | **~40** |

Adjust proportions as you progress: Phase 1 is heavier on tutorials/implementation, Phase 3 is heavier on experiments and writing.

---

## Appendix D: Self-Assessment Checkpoints

**End of Week 2 — Can you:**
- [ ] Explain the transformer forward pass from memory (residual stream view)?
- [ ] Implement scaled dot-product attention from raw matrix operations?
- [ ] Run activation patching and interpret the results?
- [ ] Train a linear probe on model activations?
- [ ] Create and apply a steering vector?

**End of Week 4 — Can you:**
- [ ] Use TransformerLens or nnsight to investigate a model's behavior on a novel prompt?
- [ ] Use SAEs to identify interpretable features and explain a model's prediction?
- [ ] Use an LLM API programmatically for dataset generation and evaluation?
- [ ] Conduct an end-to-end mini-investigation using 3+ techniques?
- [ ] Articulate the 3 most promising research directions in mech interp today?

**End of Week 7 — Can you:**
- [ ] Complete a 3–5 day research project from idea to write-up?
- [ ] Get unstuck within 2 hours using exploration techniques?
- [ ] Write a clear 1-page summary of research findings?
- [ ] Identify when a result is likely false and design experiments to test it?
- [ ] Generate 20 research ideas in one brainstorm session?

**End of Week 12 — Can you:**
- [ ] Execute a 1–2 week research sprint with clear hypotheses and experiments?
- [ ] Write a public blog post or short paper about your findings?
- [ ] Critically evaluate a mech interp paper (identify strengths, limitations, and what's missing)?
- [ ] Use literature search to contextualize your work and avoid reinventing the wheel?
- [ ] Articulate what you'd want to spend the next 3 months working on, and why?