import os
import uuid
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure no GUI backend
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment or .env file.")

client = OpenAI(api_key=api_key)

def generate_explanations(portfolio):
    """
    Generate detailed explanations, ESG averages, and chart file encodings.
    Returns (lines, avg_esg, esg_chart_b64, alloc_chart_b64, sentiment_chart_b64, captions)
    """
    if not portfolio:
        return ["No portfolio selected."], 0, None, None, None, {}

    lines = []
    weights = []

    # ESG scores and allocations
    esg_scores = [s.get("ESG_Score", 0) or 0 for s in portfolio]
    total = sum(esg_scores) if sum(esg_scores) > 0 else len(portfolio)
    allocations = [score / total for score in esg_scores]

    # ---- Stock-level rationales ----
    for stock, alloc in zip(portfolio, allocations):
        t = stock.get("Ticker", "N/A")
        esg = stock.get("ESG_Score", 0)
        esg_cat = stock.get("ESG_Category", "-")
        risk_cat = stock.get("Risk_Category", "-")
        sentiment = stock.get("Sentiment", "-")
        news_url = stock.get("NewsURL", "")
        news = f"<a href='{news_url}' target='_blank'>Click here for latest news</a>" if news_url else "No news available"

        pct = round(alloc * 100, 1)
        weights.append(pct)

        base_rationale = (
            f"<b>{t}</b> → Suggested allocation: <b>{pct}%</b><br>"
            f"• ESG Score: {esg:.1f} ({esg_cat})<br>"
            f"• Risk: {risk_cat}<br>"
            f"• Sentiment: {sentiment}<br>"
            f"• Latest News: {news}<br>"
        )

        # --- GPT-enhanced explanation ---
        try:
            prompt = f"""
You are an ESG financial advisor. Explain why {t} received {pct}% allocation.

Data:
- ESG Score: {esg:.1f} ({esg_cat})
- Risk: {risk_cat}
- Sentiment: {sentiment}
- Portfolio Avg ESG: {np.mean(esg_scores):.1f}

Provide 3–4 sentences in a professional but easy-to-read tone.
Explain why this allocation is appropriate NOW, referencing risk/market context.
"""
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an ESG financial advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=180
            )
            gpt_text = resp.choices[0].message.content.strip()
            rationale = base_rationale + f"<i>AI Insight:</i> {gpt_text}"
        except Exception as e:
            rationale = base_rationale + f"<i>AI Explanation unavailable: {e}</i>"

        lines.append(rationale)

    # ---- Portfolio insights ----
    avg_esg = round(np.mean(esg_scores), 1)
    insight = (
        f"<br><b>Portfolio Insight:</b><br>"
        f"• Average ESG score: {avg_esg}.<br>"
        f"• Allocation balances ESG, sentiment, and risk.<br>"
        f"• Diversification across industries reduces exposure.<br>"
    )
    lines.append(insight)

    # ---- Charts ----
    tickers = [s.get("Ticker", "") for s in portfolio]

    # ESG Distribution
    plt.figure(figsize=(6,4))
    plt.bar(tickers, esg_scores, color="#16a34a")
    plt.title("ESG Score Distribution")
    plt.xlabel("Ticker")
    plt.ylabel("ESG Score")
    plt.tight_layout()
    esg_path = f"static/esg_chart_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(esg_path)
    plt.close()

    # Allocation Pie
    plt.figure(figsize=(6,4))
    plt.pie(weights, labels=tickers, autopct="%1.1f%%", startangle=140)
    plt.title("Portfolio Allocation Strategy")
    plt.tight_layout()
    alloc_path = f"static/alloc_chart_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(alloc_path)
    plt.close()

    # Sentiment breakdown
    sentiments = [s.get("Sentiment", "neutral").lower() for s in portfolio]
    labels, counts = np.unique(sentiments, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar(labels, counts, color=["#16a34a","#ca8a04","#dc2626"])
    plt.title("Sentiment Breakdown")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    sent_path = f"static/sentiment_chart_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(sent_path)
    plt.close()

    # Encode to base64
    def to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    esg_chart_b64 = to_b64(esg_path)
    alloc_chart_b64 = to_b64(alloc_path)
    sentiment_chart_b64 = to_b64(sent_path)

    # ---- GPT captions for visuals ----
    captions = {}
    try:
        viz_prompt = f"""
You are an ESG analyst. Interpret these visuals for an investor who is new to the investing world:

1. ESG scores for {tickers} → {esg_scores}
2. Allocations (%) → {weights}
3. Sentiment breakdown → {dict(zip(labels, counts))}

Write short captions (3–4 sentences each) explaining what each chart suggests and go into depth to further explain how this would impact the investement they wish to make.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":viz_prompt}],
            max_tokens=200
        )
        gpt_text = resp.choices[0].message.content.strip().split("\n")
        captions["esg"] = gpt_text[0] if len(gpt_text) > 0 else "ESG chart explanation unavailable."
        captions["alloc"] = gpt_text[1] if len(gpt_text) > 1 else "Allocation chart explanation unavailable."
        captions["sentiment"] = gpt_text[2] if len(gpt_text) > 2 else "Sentiment chart explanation unavailable."
    except Exception as e:
        captions = {
            "esg": f"ESG explanation unavailable: {e}",
            "alloc": "Allocation insight unavailable.",
            "sentiment": "Sentiment insight unavailable."
        }

    return lines, avg_esg, esg_chart_b64, alloc_chart_b64, sentiment_chart_b64, captions
