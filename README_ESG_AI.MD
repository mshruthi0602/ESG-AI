This project is a prototype of an AI-powered financial advisor chatbot. It integrates ESG scoring, FinBERT sentiment analysis, and report generation to help non-technical retail investors make ESG-aligned investment choices.
Requirements
Make sure you have Python 3.9+ installed. Then install the dependencies:
pip install -r requirements.txt
Key dependencies:
- Flask – backend + chatbot interface
- transformers – FinBERT sentiment analysis
- pandas – ESG dataset handling
- beautifulsoup4, requests – news scraping
- jinja2 – HTML report rendering
-OpenAI for expalnations
Running the Application
1. Clone the repository / unzip project folder
   cd esg-ai
2. Start the Flask server
   python app.py
3. Open your browser at: http://127.0.0.1:5000
You should now see the chatbot interface.
Walkthrough
1. Enter a Query (e.g., 'medium esg medium risk in tech')
2. System Processing:
   - Filters ESG dataset by your preferences
   - Scrapes latest news headlines
   - Classifies sentiment with FinBERT
   - Ranks and returns recommendations
3. View Chatbot Results:
   - Companies, ESG scores, and sentiment appear in the chat window
   - Explanations provided (rule-based now, GPT integration in progress)
4. Download a Report:
   - Click 'Download ESG Report'
   - Generates an HTML report with input preferences, ESG scores, sentiment, AI explanations, and summary table
Testing Checklist
 Chatbot parses ESG + risk queries
 FinBERT returns correct sentiment (positive/neutral/negative)
 ESG filter works across High/Medium/Low tiers
 Report downloads successfully

• GPT explanations are integrated. 
• Cached news headlines are included for if needed for offline testing.
• Planned improvements:
  - NLP parsing for natural phrases (e.g., 'sustainable tech stocks')
  - Reinforcement learning agent for adaptive portfolio scoring
