
import streamlit as st
import requests
import yfinance as yf
import os
from google import genai
from google.genai import types

# 1. grab key from terminal so github doesn't leak it
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("🔑 API Key not found! Please run the export command in your terminal first.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. setup agent tools
def get_weather(city: str) -> str: 
    """Takes a city name as an input string and returns the current weather description for that city."""
    # using wttr.in because it doesn't need an api key
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
        # fetch data from yahoo finance
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            return f"The current stock price of {ticker.upper()} is {price:.2f}."
        # fallback just in case history block comes back empty
        price = stock.fast_info['last_price']
        return f"The current stock price of {ticker.upper()} is {price:.2f}."
    except Exception:
        return f"Could not find stock data for '{ticker}'."

# map string names to the actual python functions for the execution loop later
tools_map = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price
}

# 3. simple streamlit ui setup
st.set_page_config(page_title="Gemini Tool Agent", page_icon="♊", layout="centered")
st.title("♊ Gemini Multi-Step Agent")
st.caption("Powered by Gemini 1.5 Flash & Streamlit")

# init chat history state so messages don't wipe on rerun
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# render past chat logs
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# handle user input
if query := st.chat_input("Ask about weather, stocks, or both..."):
    st.session_state.display_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        # placeholders to update status text dynamically without breaking chat look
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        status_placeholder.markdown("🧠 *Gemini is planning execution steps...*")
        
        # pass tools to gemini so it can decide what function to call
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=query,
            config=types.GenerateContentConfig(
                tools=[get_weather, get_stock_price],
                temperature=0.0 # keep it precise so it doesn't hallucinate tools
            )
        )
        
        function_calls = response.function_calls
        
        # if gemini decided it needs a tool, run it here
        if function_calls:
            tool_outputs = []
            for call in function_calls:
                name = call.name
                args = call.args
                # grab the first argument out of the dictionary format
                first_arg = list(args.values())[0] if args else ""
                
                status_placeholder.markdown(f"🛠️ *Gemini executed tool:* `{name}` with `{first_arg}`")
                
                # safely execute matching tool from the map
                if name in tools_map:
                    result = tools_map[name](first_arg)
                    tool_outputs.append(result)
            
            # send data back to gemini so it can summarize the raw tool outputs cleanly
            status_placeholder.markdown("📝 *Compiling final summaries...*")
            final_response = client.models.generate_content(
                model='gemini-2.0-flash', # matched this to 2.0-flash to avoid old 404s
                contents=f"User asked: {query}. The tool results were: {', '.join(tool_outputs)}. Provide the final compiled answer."
            )
            answer = final_response.text
        else:
            # no tools needed, just return regular llm response
            answer = response.text
            
        # clean up status UI and print final output
        status_placeholder.empty()
        response_placeholder.write(answer)
        st.session_state.display_history.append({"role": "assistant", "content": answer})

