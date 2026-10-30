# Alo Yoga: Multimodal AI for Reducing Return-Journey Friction

A multimodal analytics and RAG (retrieval-augmented generation) pipeline that finds where customers get stuck in Alo Yoga's return journey, using text, images and video, and tests whether a RAG assistant can answer their return and sizing questions.

Built as a group project (Group 1) for the *Multimodal AI Strategy* course, PGPM, S.P. Jain Institute of Management & Research (SPJIMR), 2026. My role: led the design and build of the pipeline.

## The problem

Alo Yoga customers hit friction across the return journey: unclear eligibility, final-sale traps, no in-house alterations, return-and-rebuy inventory loss and slow refunds. The project measures that friction from real customer evidence and prototypes an assistant that reduces customer effort.

## What the notebook does

| Stage | What it covers | Methods |
| --- | --- | --- |
| Text | Reviews and evidence corpus; friction taxonomy | VADER vs transformer sentiment, aspect-based sentiment, LDA and BERTopic |
| Image | Catalog, UGC and UI screenshots | Interpretable visual features, CLIP zero-shot (evaluated against known labels), Gemini vision captioning |
| Video | YouTube and short-form creator content | Engagement analysis, timestamped transcript tagging, Gemini video understanding, latency/cost trade-off |
| Fusion | One friction map across modalities | Cross-modal corroboration and conflict analysis, Gemini few-shot classification |
| Retrieval | Search over the evidence corpus | TF-IDF vs sentence embeddings vs hybrid vs policy-routed retrieval (Hit@3, Recall@3, MRR); k-means customer segmentation |
| RAG | Return-journey assistant | Text-only RAG vs multimodal hybrid RAG, with a PII scrubber at ingestion |
| Evaluation | 13 gold queries, human-scored | Accuracy and customer-effort scores; help-page baseline |
| Governance | Responsible-AI checks | Source verification, PII handling, greenwashing check, deepfake/disclosure/copyright risks |

## Key result

On a 13-query human-scored evaluation, the **text-only RAG beat the multimodal version (3.85 vs 3.25 out of 5)**. The recommendation was a lower-cost, text-first rollout, adding image and video evidence only where it measurably helps.

## Running it

The notebook is built for Google Colab.

1. Open `Alo_Yoga_Multimodal_RAG.ipynb` in Colab.
2. Add a Gemini API key in Colab's Secrets panel (key icon) as `GEMINI_API_KEY`. No key is stored in the notebook.
3. Upload the input datasets to the runtime (see below) and run the cells in order.

Main libraries: pandas, NumPy, scikit-learn, sentence-transformers, transformers, PyTorch, BERTopic, VADER, Google Gen AI SDK, youtube-transcript-api. See `requirements.txt`.

## Data

The input datasets (team-collected reviews, a source-verified evidence corpus, and scraped Instagram/YouTube metadata) are **not included** in this repository. They contain third-party content and personal data, so they are kept private. Notebook outputs have been cleared for the same reason.

## Note

This is an academic prototype, not affiliated with or endorsed by Alo Yoga. Product and brand names belong to their owners.
