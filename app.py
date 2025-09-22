from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os, json, re, traceback
from datetime import datetime
import warnings

from src.data_loader import load_esg_data, get_unique_industries, get_unique_sectors
from src.portfolio_optimizer import optimize_portfolio
from src.news_fetcher import get_headlines_for_universe
from src.scoring import aggregate_sentiment
from src.explanation_generator import generate_explanations
from src.industry_parser import detect_filters
from src.price_data import get_prices_matrix   


warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually")
# Optional ML booster
try:
    from src.price_data import get_price_history
    from src.features import make_panel_features
    from src.model_infer import predict_proba
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

app = Flask(__name__)

# ----------------- OpenAI: lazy client so the app can boot even without a key -----------------
_openai_client = None
_openai_err = None

def _get_openai_client():
    """Return an OpenAI client or None if unavailable; never raise at import/boot."""
    global _openai_client, _openai_err
    if _openai_client is not None or _openai_err is not None:
        return _openai_client
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            _openai_err = "OPENAI_API_KEY not set"
            return None
        _openai_client = OpenAI(api_key=api_key)
        return _openai_client
    except Exception as e:
        _openai_err = f"OpenAI init error: {e}"
        return None

# Tunables
MAX_TICKERS = 80
NEWS_PER_TICKER = 5
NEWS_CACHE_MIN = 60
ENABLE_ML_BLEND = True


# ----------------- utils -----------------
def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_esg_and_risk(message: str):
    """Rule-based extraction of ESG and Risk preferences."""
    m = _norm(message)
    esg = "High" if any(k in m for k in ["high esg", "strong esg", "good esg"]) else \
          "Low" if any(k in m for k in ["low esg", "poor esg", "weak esg"]) else \
          "Medium"
    risk = "Low" if any(k in m for k in ["low risk", "conservative", "safe", "safety"]) else \
           "High" if any(k in m for k in ["high risk", "risky", "aggressive", "speculative"]) else \
           "Medium"
    return esg, risk


def gpt_chat_response(message: str):
    """Use GPT to handle natural queries outside strict ESG/Risk patterns."""
    client = _get_openai_client()
    if client is None:
        # Graceful fallback keeps UX smooth even without API
        return (
            "I can help you build an ESG portfolio. "
            "Tell me your preferences, for example: 'medium esg, low risk in technology'. "
            f"(AI assistant unavailable: {_openai_err or 'no API key'})"
        )
    try:
        prompt = f"""
You are an ESG financial advisor chatbot. A user asked: "{message}".

Your goals:
1. If the user wants portfolio advice, guide them smoothly into ESG + Risk preferences.
2. If they ask general ESG/market questions, answer fluently in 3–5 sentences.
3. Always be professional, clear, and helpful.

Reply conversationally.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an ESG financial advisor chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI assistant unavailable. Error: {e}"


# ----------------- routes -----------------
@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"),
                               "favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.route("/", methods=["GET"])
def home():
    industries = get_unique_industries()
    return render_template("chat_index.html", industries=industries)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"reply": "Please type a request.", "portfolio": []})

        # Greetings
        if message.lower() in {"hi", "hello", "hey"}:
            return jsonify({
                "reply": "👋 Hello! I’m your ESG-AI Advisor. Tell me your preferences, e.g., *medium esg medium risk in tech*, or just ask me about the best ESG portfolios.",
                "portfolio": [],
                "step": "awaiting_request"
            })

        # Try rule-based ESG/Risk parsing
        esg, risk = parse_esg_and_risk(message)
        esg_df = load_esg_data()

        # Sector/Industry detection
        dataset_sectors = get_unique_sectors()
        dataset_inds = get_unique_industries()
        display_label, sectors_keep, industries_keep = detect_filters(message, dataset_sectors, dataset_inds)

        # If no sector/industry/evidence of portfolio query → let GPT handle
        if not sectors_keep and not industries_keep and "esg" not in message.lower() and "risk" not in message.lower():
            gpt_reply = gpt_chat_response(message)
            return jsonify({
                "reply": gpt_reply,
                "portfolio": [],
                "step": "chat"
            })

        # Apply filters
        if sectors_keep:
            esg_df = esg_df[esg_df["Sector"].str.lower().isin([s.lower() for s in sectors_keep])]
        if industries_keep:
            esg_df = esg_df[esg_df["Industry"].isin(industries_keep)]

        if esg_df.empty:
            return jsonify({
                "reply": f"❌ No companies matched **{display_label or 'All'}**. Want to try another ESG/Risk profile?",
                "portfolio": [{"Ticker": "No suitable match"}],
                "step": "restart"
            })

        # ------------ Live news → FinBERT majority vote ------------
        tickers = esg_df["Ticker"].tolist()[:MAX_TICKERS]
        headlines_by_ticker = get_headlines_for_universe(
            tickers, max_items=NEWS_PER_TICKER, use_cache_minutes=NEWS_CACHE_MIN
        )

        agg = aggregate_sentiment(headlines_by_ticker)
        sentiment_scores = {t: v["label"] for t, v in agg.items()}

        # ----------------- Optional ML booster -----------------
        ml_scores = {}
        if ENABLE_ML_BLEND and ML_AVAILABLE:
            try:
                prices = get_price_history(tickers, period="2y", interval="1d")
                feat, _ = make_panel_features(prices, esg_df, sentiment_scores)
                if feat is not None and not feat.empty:
                    ml_scores = predict_proba(feat)
            except Exception:
                pass

        # ----------------- Optimize portfolio -----------------
        all_recs = optimize_portfolio(esg_df, sentiment_scores, esg, risk, display_label)

        # Attach metadata
        for row in all_recs:
            t = row.get("Ticker")
            if not t or t == "No suitable match":
                continue
            if t in agg:
                row["News"] = agg[t]["top_headline"]
                row["NewsURL"] = agg[t].get("url", f"https://www.bing.com/news/search?q={t}+stock")
                row["Sentiment"] = agg[t]["label"]
                row["SentimentConfidence"] = agg[t]["confidence"]
            if ml_scores:
                row["ML_Score"] = round(float(ml_scores.get(t, 0)), 3)

        # Tier separation
        greens = [r for r in all_recs if r.get("Tier") == "Green"]
        yellows = [r for r in all_recs if r.get("Tier") == "Yellow"]
        reds = [r for r in all_recs if r.get("Tier") == "Red"]

        if greens:
            reply = f"✅ Found **{len(greens)} Green matches** (exact ESG & Risk). Please pick up to 3 for your report."
            shown = greens
            step = "pick_tickers"
        elif yellows or reds:
            reply = "⚠️ No exact matches. Here are the closest alternatives (Yellow/Red)."
            shown = yellows + reds
            step = "pick_tickers"
        else:
            reply = "❌ No usable data. Please try another ESG/Risk profile."
            shown = [{"Ticker": "No suitable match"}]
            step = "restart"

        return jsonify({
            "reply": reply,
            "portfolio": shown,
            "step": step,
            "esg_preference": esg,
            "risk_tolerance": risk,
            "industry": display_label
        })

    except Exception as e:
        print(f"[ERROR] /chat → {e}")
        traceback.print_exc()
        return jsonify({"reply": f"❌ An error occurred: {e}", "portfolio": [], "step": "restart"})


@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        payload = request.form
        portfolio = json.loads(payload["portfolio_json"])
        explanation = payload.get("explanation", "")
        esg = payload.get("esg_preference", "Medium")
        risk = payload.get("risk_tolerance", "Medium")
        industry = payload.get("industry", "")

        # Human-readable explanations + visuals
        lines, avg_esg, esg_chart, alloc_chart, sentiment_chart, captions = generate_explanations(portfolio)

        explanations_html = "<br><br>".join(lines)

        rendered = render_template(
            "report_template.html",
            portfolio=portfolio,
            explanation=explanation,
            esg_preference=esg,
            risk_tolerance=risk,
            industry=industry,
            gpt_explanation=explanations_html,
            avg_esg=avg_esg,
            esg_chart=esg_chart,
            alloc_chart=alloc_chart,
            sentiment_chart=sentiment_chart,
            captions=captions,
            generated_on=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

        # ensure 'static' exists before writing
        os.makedirs("static", exist_ok=True)

        filename = f"esg_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        path = os.path.join("static", filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(rendered)

        return send_file(path, as_attachment=True, download_name=filename, mimetype="text/html")
    except Exception as e:
        print(f"[ERROR] /download_report → {e}")
        traceback.print_exc()
        return f"Error generating report: {e}", 500


@app.route("/health", methods=["GET"])
def health():
    """Simple health check to verify the server is running."""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    print("Device set to use: cpu")
    # Flask's reloader can spawn twice; set use_reloader=False if you see duplicate logs
    app.run(debug=True, use_reloader=True)
