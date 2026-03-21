---
name: research-paper
description: "Research a paper or topic. Use when: user wants to read a paper, explore a concept, download a reference, or write an explainer. Downloads the paper, analyses it, writes an explainer, and correlates with the project plan."
---

# Research Paper

Download, read, and analyse a paper or topic relevant to the QAOA-XORSAT project. Write an explainer and correlate findings with the project plan.

## When to Use

- User provides an arXiv ID or paper title
- User asks about a quantum computing concept relevant to the project
- User says "research", "read this paper", "what does X mean"

## Procedure

### Step 1 — Identify the Paper

If the user provides an arXiv ID (e.g. `2503.12789`):

```bash
# Download from arXiv
curl -fsSL -o ".project/papers/${ARXIV_ID}-${SHORT_NAME}.pdf" \
  "https://arxiv.org/pdf/${ARXIV_ID}.pdf"
```

If the user provides a title or topic, search for the relevant paper.

### Step 2 — Check Existing Knowledge

Read existing explainers in `.project/learning/` to avoid duplication:

```bash
ls -1 .project/learning/*.md
```

Check if this paper is already covered or referenced.

### Step 3 — Analyse the Paper

Read the paper and identify:
- **Key contributions**: What new results or methods does it introduce?
- **Methods**: What mathematical/computational techniques are used?
- **Results**: What are the main quantitative findings?
- **Relevance**: How does this relate to our QAOA-XORSAT computation?
- **Impact on our approach**: Does this change anything in `.project/PLAN.md`?

### Step 4 — Write an Explainer

Create `.project/learning/NN-explainer-{topic}.md` where NN is the next number:

```markdown
# Explainer: {Paper Title or Topic}

**Paper**: {authors} ({year}) — {title} — arXiv:{id}

## Why This Matters for Us
1-2 paragraphs on relevance to QAOA on D-regular Max-k-XORSAT.

## Key Ideas
Explain the core contributions at PhD student level.

## Technical Details
Mathematics and methods, with concrete examples at (k=3, D=4) where possible.

## Implications for Our Project
- What this means for our implementation approach
- Any new validation targets
- Changes to complexity estimates or feasibility
```

### Step 5 — Update Project Context

1. Add a dated entry to `.project/journal.md` summarising the finding
2. If the paper affects the plan, suggest specific updates to `.project/PLAN.md`
3. If the paper provides new comparison data, note it for `04-our-problem.md`

### Step 6 — Report to User

Summarise:
- What the paper contributes
- How it affects our project
- Recommended next steps
