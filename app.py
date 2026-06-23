import streamlit as st
import requests
import yfinance as yf
import os
from google import genai
from google.genai import types

# 1. SAFELY FETCH THE KEY FROM THE TERMINAL ENVIRONMENT
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("🔑 API Key not found! Please run the export command in your terminal first.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. DEFINE NATIVE AGENT TOOLS
def get_weather(city: str) -> str: 
    """Takes a city name as an input string and returns the current weather description for that city."""
    url = f"https://wttr.in/{city}?format=%C+%t" 
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text}."
        return "Weather data temporarily unavailable."
    except Exception:
        return "Could not connect to weather service."

def get_stock_price(ticker: str) -> str:
    """Takes an official stock market ticker symbol (e.g., 'AAPL', 'TCS.NS') and returns its live current trading price."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            return f"The current stock price of {ticker.upper()} is {price:.2f}."
        price = stock.fast_info['last_price']
        return f"The current stock price of {ticker.upper()} is {price:.2f}."
    except Exception:
        return f"Could not find stock data for '{ticker}'."

tools_map = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price
}

# 3. STREAMLIT INTERFACE
st.set_page_config(page_title="Gemini Tool Agent", page_icon="♊", layout="centered")
st.title("♊ Gemini Multi-Step Agent")
st.caption("Powered by Gemini 1.5 Flash & Streamlit")

if "display_history" not in st.session_state:
    st.session_state.display_history = []

for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask about weather, stocks, or both..."):
    st.session_state.display_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        status_placeholder.markdown("🧠 *Gemini is planning execution steps...*")
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=query,
            config=types.GenerateContentConfig(
                tools=[get_weather, get_stock_price],
                temperature=0.0
            )
        )
        
        function_calls = response.function_calls
        
        if function_calls:
            tool_outputs = []
            for call in function_calls:
                name = call.name
                args = call.args
                first_arg = list(args.values())[0] if args else ""
                
                status_placeholder.markdown(f"🛠️ *Gemini executed tool:* `{name}` with `{first_arg}`")
                
                if name in tools_map:
                    result = tools_map[name](first_arg)
                    tool_outputs.append(result)
            
            status_placeholder.markdown("📝 *Compiling final summaries...*")
            final_response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=f"User asked: {query}. The tool results were: {', '.join(tool_outputs)}. Provide the final compiled answer."
            )
            answer = final_response.text
        else:
            answer = response.text
            
        status_placeholder.empty()
        response_placeholder.write(answer)
        st.session_state.display_history.append({"role": "assistant", "content": answer})