# chatbot.py (fixed & runnable)
# -*- coding: utf-8 -*-
import os
import json
import getpass
import operator
import base64
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from typing_extensions import Literal
from io import BytesIO
import streamlit as st
# Data modeling
from pydantic import BaseModel, Field

# Image processing
from PIL import Image
import requests
# IPython display imports are harmless if present but not required at runtime for Streamlit
try:
    from IPython.display import Image as IPImage, display, Markdown
except Exception:
    IPImage = None
    display = None
    Markdown = None

# NOTE: we use the Groq SDK directly to avoid mixing incompatible wrappers
from groq import Groq

# Keep your original langchain_core imports for StructuredTool/@tool usage
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.tools import tool, StructuredTool
from langchain_core import tools

# Financial data
import yfinance as yf
import pandas as pd

# Web search and HTML parsing
from bs4 import BeautifulSoup
from urllib.request import urlopen
from tavily import TavilyClient

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import plotly.io as pio

# YouTube API
import googleapiclient.discovery
import googleapiclient.errors

# State graph
from langgraph.graph import StateGraph, START, END

# Environment variables
from dotenv import load_dotenv

# OpenAI (used only for optional translation SUTRA calls if configured)
from openai import OpenAI

# optional NVIDIA endpoints (kept as comments / fallback)
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

# ---------------------------
# LLM shim using Groq SDK
# ---------------------------
# This shim gives a minimal-compatible interface for:
# - llm.invoke(messages_or_string)
# - llm.predict(prompt)
# - llm.bind_tools(... ) -> returns self (no-op)
# - llm.with_structured_output(Model) -> returns wrapper with invoke(...) that parses JSON -> model
#
# It uses groq.Groq client for chat completions under the hood.
GROQ_API_KEY = None
try:
    GROQ_API_KEY = st.secrets.REST.GROQ_API_KEY
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    # We'll still allow the module to import; runtime calls will error clearly
    st.warning("Warning: GROQ API key not found in streamlit secrets or environment. Add it to .streamlit/secrets.toml as REST.GROQ_API_KEY.")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Choose Mixtral model per user's choice (Option A)
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

class LLMShimResponse:
    def __init__(self, content: str):
        self.content = content

class LLMShim:
    def __init__(self, model_name: str = DEFAULT_GROQ_MODEL, client_instance: Optional[Groq] = client):
        self.model_name = model_name
        self.client = client_instance

    def _messages_to_text(self, messages):
        # Accept strings, or lists of SystemMessage/HumanMessage/ToolMessage/AIMessage or raw dicts
        if isinstance(messages, str):
            return messages
        if messages is None:
            return ""
        # If it's a list of message objects
        if isinstance(messages, (list, tuple)):
            parts = []
            for m in messages:
                if isinstance(m, (SystemMessage, HumanMessage, AIMessage, ToolMessage)):
                    parts.append(m.content)
                elif isinstance(m, dict):
                    parts.append(m.get("content", ""))
                else:
                    parts.append(str(m))
            return "\n".join(parts)
        # If it's a single message object
        if isinstance(messages, (SystemMessage, HumanMessage, AIMessage, ToolMessage)):
            return messages.content
        # fallback
        return str(messages)

    def invoke(self, messages, max_tokens: int = 1024, temperature: float = 0.2):
        text = self._messages_to_text(messages)
        if not self.client:
            raise ValueError("Groq client not initialized. Set GROQ_API_KEY in streamlit secrets or environment.")
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )

            # ALWAYS correct for Groq:
            # resp.choices[0].message.content
            content = ""
            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]

                # If message is object → normal
                if hasattr(choice.message, "content"):
                    content = choice.message.content

                # If message is dict → fallback
                elif isinstance(choice.message, dict):
                    content = choice.message.get("content", "")

                else:
                    content = str(choice)

            else:
                content = str(resp)

            return LLMShimResponse(content=content)

        except Exception as e:
            raise RuntimeError(f"Groq SDK error: {e}")


    def predict(self, prompt: str):
        return self.invoke(prompt).content

    def bind_tools(self, tools_list, tool_choice='auto'):
        # no-op shim for compatibility
        return self

    def with_structured_output(self, model_cls):
        # Return a wrapper object providing invoke(...) that parses JSON content and returns a pydantic model
        parent = self
        class Wrapper:
            def __init__(self, parent, model_cls):
                self.parent = parent
                self.model_cls = model_cls
            def invoke(self, prompt_or_messages):
                r = self.parent.invoke(prompt_or_messages)
                text = r.content.strip()
                # Try to interpret as JSON and parse into pydantic model_cls
                try:
                    parsed = json.loads(text)
                    return self.model_cls.parse_obj(parsed)
                except Exception:
                    # Fallback: try to extract a simple string field if model_cls expects 'step'
                    try:
                        return self.model_cls.parse_obj({"step": text})
                    except Exception:
                        # Last fallback: return object with attribute 'step' if possible
                        class Fallback:
                            def __init__(self, content):
                                self.step = content
                        return Fallback(text)
        return Wrapper(parent, model_cls)

# instantiate shim
llm = LLMShim()

# For backward-compatibility aliases used elsewhere
llm_normal = llm

# ---------------------------
# Prompt templates (unchanged)
# ---------------------------
query_writer_instruction_web = """Your goal is to generate a targeted web search query related to financial investments or any finance-related topic specified by the user.

<TOPIC>
{finance_topic}
</TOPIC>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "aspect": The specific aspect of the finance topic being researched
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "best index funds for long-term investment 2025",
    "aspect": "investment strategy",
    "rationale": "Identifying top-performing index funds for long-term portfolio growth"
}}
</EXAMPLE>

Provide your response in JSON format:
"""

summarizer_instruction_web = """<GOAL>
Generate a high-quality summary of the web search results, focusing on financial investments or the specific finance-related topic requested by the user.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant financial insights, trends, or strategies from the search results.
2. Ensure a coherent flow of information while keeping it concise and actionable.

When EXTENDING an existing summary:
1. Read the existing summary and new search results carefully.
2. Compare the new information with the existing summary.
3. For each piece of new information:
    a. If it builds on an existing point, integrate it smoothly.
    b. If it introduces a new relevant aspect, add a separate paragraph.
    c. If it’s irrelevant to financial investments, ignore it.
4. Ensure all additions align with the user’s finance-related query.
5. Verify that the final output differs from the original summary while improving its depth.

<FORMATTING>
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.
</FORMATTING>
"""

reflection_instructions_web = """You are an expert financial research assistant analyzing a summary about {finance_topic}.

<GOAL>
1. Identify missing details or areas that need deeper exploration.
2. Generate a follow-up question to help expand financial knowledge.
3. Focus on investment strategies, market trends, risk factors, regulations, or financial instruments that weren’t fully covered.
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and provides necessary context for a web search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- "knowledge_gap": Describe what financial information is missing or unclear.
- "follow_up_query": Write a specific question to address this gap.
</FORMAT>
"""

# ---------------------------
# State type
# ---------------------------
class State(TypedDict):
    route: Literal['Web_query', 'Normal_query', 'Financial_Analysis', 'YouTube_Recommender', 'Plot_Graph']
    research_topic: str
    search_query: str
    web_research_results: List[str]
    sources_gathered: List[str]
    research_loop_count: int
    running_summary: str
    image: list[str]
    image_processed: bool
    messages: List[Any]
    original_messages: List[Any]
    plot_type: Optional[str]
    ticker: Optional[str]
    plot_json: Optional[str]

# ---------------------------
# Financial data helpers
# ---------------------------
def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def fetch_balance(ticker, tp="Annual"):
    ticker_obj = yf.Ticker(ticker)
    bs = ticker_obj.balance_sheet if tp == "Annual" else ticker_obj.quarterly_balance_sheet
    # drop columns where mostly NaN
    return bs.loc[:, bs.isna().mean() < 0.5]

# ---------------------------
# Plotting helpers (unchanged)
# ---------------------------
def plot_candles_stick(df, title=""):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.update_layout(title=title)
    return fig

def plot_balance(df, ticker="", currency=""):
    df.columns = pd.to_datetime(df.columns).strftime('%b %d, %Y')
    components = {
        'Total Assets': {'color': 'forestgreen', 'name': 'Assets'},
        'Stockholders Equity': {'color': 'CornflowerBlue', 'name': "Stockholder's Equity"},
        'Total Liabilities Net Minority Interest': {'color': 'tomato', 'name': "Total Liabilities"},
    }

    fig = go.Figure()
    for component in components:
        if component in df.index:
            if component == 'Total Assets':
                fig.add_trace(go.Bar(
                    x=[df.columns, ['Assets'] * len(df.columns)],
                    y=df.loc[component],
                    name=components[component]['name'],
                    marker=dict(color=components[component]['color'])
                ))
            else:
                fig.add_trace(go.Bar(
                    x=[df.columns, ['L+E'] * len(df.columns)],
                    y=df.loc[component],
                    name=components[component]['name'],
                    marker=dict(color=components[component]['color'])
                ))

    # annotations
    try:
        offset = 0.03 * df.loc['Total Assets'].max()
        for i, date in enumerate(df.columns):
            fig.add_annotation(
                x=[date, "Assets"],
                y=df.loc['Total Assets', date] / 2,
                text=str(round(df.loc['Total Assets', date] / 1e9, 1)) + 'B',
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center"
            )
            percentage = round((df.loc['Total Liabilities Net Minority Interest', date] / df.loc['Total Assets', date]) * 100, 1)
            fig.add_annotation(
                x=[date, "L+E"],
                y=df.loc['Stockholders Equity', date] + df.loc['Total Liabilities Net Minority Interest', date] / 2,
                text=str(percentage) + '%',
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center"
            )
            if i > 0:
                percentage = round((df.loc['Total Assets'].iloc[i] / df.loc['Total Assets'].iloc[i - 1] - 1) * 100, 1)
                sign = '+' if percentage >= 0 else ''
                fig.add_annotation(
                    x=[date, "Assets"],
                    y=df.loc['Total Assets', date] + offset,
                    text=sign + str(percentage) + '%',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
    except Exception:
        # ignore annotation errors for malformed frames
        pass

    fig.update_layout(
        barmode='stack',
        title=f'Accounting Balance: {ticker}',
        xaxis_title='Year',
        yaxis_title=f'Amount (in {currency})',
        legend_title='Balance components',
    )
    return fig

def plot_assets(df, ticker="", currency=""):
    assets = {
        'Current Assets': {
            'Cash Cash Equivalents And Short Term Investments': {},
            'Receivables': {},
            'Prepaid Assets': None,
            'Inventory': {},
            'Hedging Assets Current': None,
            'Other Current Assets': None
        },
        'Total Non Current Assets': {
            'Net PPE': {},
            'Goodwill And Other Intangible Assets': {},
            'Investments And Advances': {},
            'Investment Properties': None,
            'Other Non Current Assets': None
        }
    }

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=['Current Assets', 'Non-Current Assets']
    )

    colors = pc.sequential.Blugrn[::-1]
    i = 0
    for component in assets['Current Assets']:
        if component in df.index:
            fig.add_trace(go.Bar(
                x=df.columns,
                y=df.loc[component],
                name=component,
                marker=dict(color=colors[i % len(colors)]),
                legendgroup='Current Assets',
                showlegend=True
            ), row=1, col=1)
            i += 1

    colors = pc.sequential.Purp[::-1]
    i = 0
    for component in assets['Total Non Current Assets']:
        if component in df.index:
            fig.add_trace(go.Bar(
                x=df.columns,
                y=df.loc[component],
                name=component,
                marker=dict(color=colors[i % len(colors)]),
                legendgroup='Non-current Assets',
                showlegend=True
            ), row=1, col=2)
            i += 1

    try:
        offset = 0.03 * max(df.loc['Current Assets'].max(), df.loc['Total Non Current Assets'].max())
        for i, date in enumerate(df.columns):
            fig.add_annotation(
                x=date,
                y=df.loc['Current Assets', date] + offset,
                text=str(round(df.loc['Current Assets', date] / 1e9, 1)) + 'B',
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center",
                row=1, col=1
            )
            fig.add_annotation(
                x=date,
                y=df.loc['Total Non Current Assets', date] + offset,
                text=str(round(df.loc['Total Non Current Assets', date] / 1e9, 1)) + 'B',
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center",
                row=1, col=2
            )
    except Exception:
        pass

    fig.update_layout(
        barmode='stack',
        title=f'Assets: {ticker}',
        xaxis1=dict(title='Date', type='date', tickvals=df.columns),
        xaxis2=dict(title='Date', type='date', tickvals=df.columns),
        yaxis_title=f'Amount (in {currency})',
        legend_title='Asset Components',
    )
    return fig

# ---------------------------
# Ticker retrieval tool
# ---------------------------
import logging
from typing import Optional, Dict

class TickerRetrievalTool:
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the Ticker Retrieval Tool

        :param logger: Optional logger for tracking tool operations
        """
        self.logger = logger or logging.getLogger(__name__)
        self.yfinance_url = "https://query2.finance.yahoo.com/v1/finance/search"
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'

    def get_ticker(self, company_name: str, country: str) -> Optional[str]:
        """
        Retrieve stock ticker for a given company and country

        :param company_name: Name of the company to search
        :param country: Country of the stock exchange
        :return: Stock ticker symbol or None if not found
        """
        try:
            params = {
                "q": company_name,
                "quotes_count": 5,
                "country": country
            }

            response = requests.get(
                url=self.yfinance_url,
                params=params,
                headers={'User-Agent': self.user_agent},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if country and country.upper() == 'INDIA':
                for quote in data.get('quotes', []):
                    if quote.get('exchange') == 'NSI':
                        return quote.get('symbol')
            else:
                quotes = data.get('quotes', [])
                if quotes:
                    return quotes[0].get('symbol')
            return None
        except requests.RequestException as e:
            self.logger.error(f"Network error retrieving ticker: {e}")
            return None
        except (KeyError, IndexError) as e:
            self.logger.error(f"No ticker found for {company_name} in {country}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in ticker retrieval: {e}")
            return None

    def tool_description(self) -> Dict[str, Any]:
        return {
            "name": "stock_ticker_retrieval",
            "description": "Retrieves stock ticker symbols for companies across different countries",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Full or partial name of the company"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country of the stock exchange (e.g., 'India', 'US')"
                    }
                },
                "required": ["company_name", "country"]
            }
        }

# ---------------------------
# Company & country extraction via LLM
# ---------------------------
def get_company_country(query):
    prompt_get_country_company = """You are an expert at extracting precise company and country information from user queries. Follow these guidelines carefully:

Extraction Rules:
1. Identify the specific company name mentioned in the query
2. Determine the country of origin for the identified company
3. For index-related queries, use these specific mappings:
- NIFTY50 → ['^NSEI', 'India']
- NIFTY100 → ['^CNX100', 'India']
- NIFTY MIDCAP 150 → ['NIFTYMIDCAP150.NS', 'India']

Output Format Requirements:
- Always respond in a strict JSON format
- Use double quotes for keys and string values
- Ensure no trailing commas
- Keys must be exactly: "company" and "country"

Example Outputs:
- For "Tell me about Apple":
{"company": "Apple Inc.", "country": "United States"}
- For "NIFTY50 performance":
{"company": "^NSEI", "country": "India"}

Your task: Extract the company and country from the following query:
"""
    final_prompt = prompt_get_country_company + query
    # Use llm shim
    resp = llm.invoke(final_prompt)
    try:
        data = json.loads(resp.content)
        company = data.get('company')
        country = data.get('country')
        return [company, country]
    except Exception:
        # Fallback heuristic: very simple parsing
        text = query.strip()
        # if 'nifty' mention
        if 'nifty' in text.lower():
            if '50' in text:
                return ["^NSEI", "India"]
            if '100' in text:
                return ["^CNX100", "India"]
            return ["^NSEI", "India"]
        words = text.split()
        return [words[-1], ""] if words else [query, ""]

# ---------------------------
# Parse / Generate / Format nodes (unchanged behaviour)
# ---------------------------
def parse_query(state: State) -> State:
    query = state["research_topic"].lower()
    data = get_company_country(query)
    company, country = data
    ticker_tool = TickerRetrievalTool()
    ticker = ticker_tool.get_ticker(company, country)
    if "candlestick" in query:
        return {"plot_type": "candlestick", "ticker": ticker}
    elif "balance" in query:
        return {"plot_type": "balance", "ticker": ticker}
    elif "assets" in query:
        return {"plot_type": "assets", "ticker": ticker}
    else:
        return {"plot_type": None, "ticker": None}

def generate_plot(state: State) -> State:
    if not state.get("plot_type") or not state.get("ticker"):
        return {"response": "I can generate candlestick charts, balance sheets, or assets visualizations. Please specify what you'd like to see (e.g., 'Show me a candlestick chart for AAPL')"}
    ticker = state["ticker"]
    plot_type = state["plot_type"]
    try:
        if plot_type == "candlestick":
            df = fetch_stock_data(ticker)
            fig = plot_candles_stick(df, title=f"{ticker} Candlestick Chart")
        elif plot_type == "balance":
            df = fetch_balance(ticker)
            fig = plot_balance(df, ticker=ticker, currency="INR")
        elif plot_type == "assets":
            df = fetch_balance(ticker)
            fig = plot_assets(df, ticker=ticker, currency="INR")
        plot_json = fig.to_json()
        return {"plot_json": plot_json}
    except Exception as e:
        return {"response": f"Error generating plot: {str(e)}"}

def format_response(state: State) -> State:
    if state.get("plot_json"):
        description = f"Here is the {state['plot_type']} plot for {state['ticker']}"
        return {"running_summary": description, "plot_json": state["plot_json"]}
    elif state.get("response"):
        return {"running_summary": state["response"]}
    else:
        return {"running_summary": "Something went wrong while processing your request"}

# ---------------------------
# create_initial_state
# ---------------------------
def create_initial_state(user_query: str, image: list[str] = []) -> State:
    return {
        "route": None,
        "research_topic": user_query,
        "search_query": "",
        "web_research_results": [],
        "sources_gathered": [],
        "research_loop_count": 0,
        "running_summary": "",
        "image": image,
        "image_processed": False,
        "messages": [HumanMessage(content=user_query)],
        "original_messages": [HumanMessage(content=user_query)],
        "plot_type": None,
        "ticker": None,
        "plot_json": None
    }

# ---------------------------
# Router first step model
# ---------------------------
class Route_First_Step(BaseModel):
    step: Literal['Web_query', 'Normal_query', 'Financial_Analysis', 'YouTube_Recommender', 'Plot_Graph'] = Field(
        None,
        description="Route the user's query based on intent and keywords."
    )

# ---------------------------
# Configuration
# ---------------------------
class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

@dataclass(kw_only=True)
class Configuration:
    max_web_research_loops: int = 3
    search_api: SearchAPI = SearchAPI.TAVILY
    fetch_full_page: bool = False
    ollama_base_url: str = "http://localhost:11434/"

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: configurable.get(f.name)
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# ---------------------------
# Web research helpers
# ---------------------------
@traceable
def tavily_search(query, include_raw_content=True, max_results=3):
    api_key = None
    try:
        api_key = st.secrets.REST.TAVILY_API_KEY
    except Exception:
        api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    tavily_client = TavilyClient(api_key=api_key)
    return tavily_client.search(query, max_results=max_results, include_raw_content=include_raw_content)

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    if isinstance(search_response, dict):
        sources_list = search_response.get('results', [])
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    unique_sources = {}
    for source in sources_list:
        url = source.get('url')
        if url and url not in unique_sources:
            unique_sources[url] = source
    formatted_text = "Sources:\n\n"
    for source in unique_sources.values():
        formatted_text += f"Source {source.get('title','No title')}:\n===\n"
        formatted_text += f"URL: {source.get('url','')}\n===\n"
        formatted_text += f"Most relevant content from source: {source.get('content','')}\n===\n"
        if include_raw_content:
            char_limit = max_tokens_per_source * 4
            raw_content = source.get('raw_content', '') or ''
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
    return formatted_text.strip()

def format_sources(search_results):
    return '\n'.join(
        f"* {source.get('title','No title')} : {source.get('url','')}"
        for source in search_results.get('results', [])
    )

def generate_query(state: State, config: RunnableConfig):
    prompt = query_writer_instruction_web.format(finance_topic=state["research_topic"]) + "\nGenerate a query for web search:"
    result = llm.invoke(prompt)
    output_text = result.content.strip()
    try:
        query_data = json.loads(output_text)
        return {"search_query": query_data['query']}
    except (json.JSONDecodeError, KeyError):
        return {"search_query": f"comprehensive analysis of {state['research_topic']}"}

def web_research(state: State, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    search_api = configurable.search_api.value if isinstance(configurable.search_api, Enum) is False else configurable.search_api.value
    if search_api == "tavily":
        search_results = tavily_search(state["search_query"], include_raw_content=True, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")
    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state["research_loop_count"] + 1,
        "web_research_results": [search_str]
    }

def summarize_sources(state: State, config: RunnableConfig):
    existing_summary = state.get('running_summary', '')
    most_recent_web_research = state.get('web_research_results', [''])[ -1]
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state['research_topic']} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state['research_topic']} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )
    prompt = summarizer_instruction_web + "\n" + human_message_content
    result = llm.invoke(prompt)
    running_summary = result.content
    # strip any <think> tokens if present
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]
    return {"running_summary": running_summary}

def reflect_on_summary(state: State, config: RunnableConfig):
    prompt = reflection_instructions_web.format(finance_topic=state['research_topic']) \
             + "\nIdentify a knowledge gap and generate a follow-up web search query based on our existing knowledge: " \
             + state.get('running_summary', '')
    result = llm.invoke(prompt)
    output_text = result.content.strip()
    try:
        follow_up_query = json.loads(output_text)
    except json.JSONDecodeError:
        follow_up_query = {"follow_up_query": f"Tell me more about {state['research_topic']}"}
    query = follow_up_query.get('follow_up_query')
    if not query:
        return {"search_query": f"Tell me more about {state['research_topic']}"}
    return {"search_query": query}

def finalize_summary(state: State):
    all_sources = "\n".join(source for source in state.get('sources_gathered', []))
    final_summary = f"{state.get('running_summary','')}\n\n### Sources:\n{all_sources}"
    final_message = HumanMessage(content=final_summary)
    return {
        "running_summary": final_summary,
        "messages": [final_message],
        "original_messages": state["original_messages"]
    }

def route_research(state: State, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    configurable = Configuration.from_runnable_config(config)
    if state.get('research_loop_count', 0) < configurable.max_web_research_loops:
        return "web_research"
    return "finalize_summary"

# ---------------------------
# Finance tools (yfinance) - ensure docstrings exist
# ---------------------------
@tool
def company_address(ticker: str) -> str:
    """Return company address for the given ticker using yfinance (falls back to a message)."""
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    parts = [info.get(k, "") for k in ['address1', 'city', 'state', 'zip', 'country']]
    return " ".join([p for p in parts if p])

@tool
def fulltime_employees(ticker: str) -> int:
    """Return full-time employee count for the given ticker using yfinance."""
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info.get('fullTimeEmployees', 0)

@tool
def last_close_price(ticker: str) -> float:
    """Return the last close price for the given ticker using yfinance history."""
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period="5d")
    if hist is not None and not hist.empty:
        return float(hist['Close'].iloc[-1])
    return 0.0

@tool
def EBITDA(ticker: str) -> float:
    """Return EBITDA for the given ticker using yfinance."""
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info.get('ebitda', 0.0)

@tool
def total_debt(ticker: str) -> float:
    """Return total debt for the given ticker using yfinance."""
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info.get('totalDebt', 0.0)

@tool
def total_revenue(ticker: str) -> float:
    """Return total revenue for the given ticker using yfinance."""
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info.get('totalRevenue', 0.0)

@tool
def debt_to_equity_ratio(ticker: str) -> float:
    """Return debt-to-equity ratio for the given ticker using yfinance."""
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info.get('debtToEquity', 0.0)

finance_tools = [
    company_address,
    fulltime_employees,
    last_close_price,
    EBITDA,
    total_debt,
    total_revenue,
    debt_to_equity_ratio
]
finance_tool_map = {t.name: t for t in finance_tools}

# ---------------------------
# LLM-driven pipelines (normal & financial)
# ---------------------------
normal_query_prompt = """
You are a financial analyst. Please answer the user's question based on what you know, don't make up anything. Make sure you elaborate it in such manner that it is easily understandable by anyone.
"""

def answer_normal_query(state: State):
    messages = state.get('messages', [])
    system_message = SystemMessage(content=normal_query_prompt + "\nFormat your response in Markdown.")
    response = llm.invoke([system_message] + messages)
    markdown_response = f"{response.content}"
    return {
        "running_summary": markdown_response,
        "messages": [HumanMessage(content=markdown_response)],
        "original_messages": state.get("original_messages", [])
    }

# Bind tools: shim returns self (tools executed later via take_action)
llm_financial_analysis = llm.bind_tools(finance_tools, tool_choice='auto')
financial_analysis_prompt = "You are a financial analyst. You are given tools for accurate data."

def call_llm(state: State):
    messages = state.get('messages', [])
    system_prompt = financial_analysis_prompt + "\nFormat your response in Markdown."
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

def exists_action(state: State):
    result = state['messages'][-1]
    # Many SDK responses won't include tool_calls in the shim; default to False
    return bool(getattr(result, "tool_calls", False))

def take_action(state: State):
    # This function expects state['messages'][-1].tool_calls to exist (LangChain-style).
    # If tool_calls are not present, we return an explanatory message.
    last_message = state['messages'][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    tool_results = []
    if not tool_calls:
        return {'messages': [HumanMessage(content="No tool calls present." )], 'running_summary': "No tool calls executed."}
    for t in tool_calls:
        try:
            tool_func = finance_tool_map[t['name']]
            result = tool_func.invoke(t['args'])
        except KeyError:
            result = f"Error: Tool {t['name']} not found"
        except Exception as e:
            result = f"Error executing tool: {str(e)}"
        # Use ToolMessage from langchain_core.messages
        tool_results.append(ToolMessage(tool_call_id=t.get('id',''), name=t['name'], content=str(result)))
    markdown_output = "\n\n"
    for result in tool_results:
        markdown_output += f"### {result.name.replace('_', ' ').title()}\n\n{result.content}\n\n"
    return {'messages': tool_results, 'running_summary': markdown_output}

def format_financial_analysis(state: State):
    messages = state.get('messages', [])
    tool_results = [msg for msg in messages if isinstance(msg, ToolMessage)]
    if tool_results:
        markdown_output = "\n\n"
        for result in tool_results:
            markdown_output += f"### {result.name.replace('_', ' ').title()}\n\n{result.content}\n\n"
    else:
        markdown_output = f"\n\n{messages[-1].content if messages else ''}"
    return {"running_summary": markdown_output, "messages": [HumanMessage(content=markdown_output)]}

# ---------------------------
# YouTube recommender (unchanged)
# ---------------------------
class YouTubeVideoRecommender:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    def get_channel_id(self, channel_name):
        request = self.youtube.search().list(
            part="snippet",
            q=channel_name,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        if response.get('items'):
            return response['items'][0]['id']['channelId']
        return None

    def search_videos_in_channel(self, channel_id, query, max_results=10):
        request = self.youtube.search().list(
            part="snippet",
            channelId=channel_id,
            q=query,
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        videos = []
        for item in response.get('items', []):
            video_id = item['id'].get('videoId')
            title = item['snippet'].get('title')
            description = item['snippet'].get('description')
            published_at = item['snippet'].get('publishedAt')
            thumbnail = item['snippet']['thumbnails']['high']['url'] if item['snippet'].get('thumbnails') else ""
            channel_title = item['snippet'].get('channelTitle')
            videos.append({
                'video_id': video_id,
                'title': title,
                'description': description,
                'published_at': published_at,
                'thumbnail': thumbnail,
                'channel': channel_title,
                'url': f"https://www.youtube.com/watch?v={video_id}"
            })
        return videos

    def recommend_videos(self, query, channels, videos_per_channel=5):
        all_videos = []
        for channel in channels:
            if isinstance(channel, str) and channel.startswith('UC') and len(channel) == 24:
                channel_id = channel
            else:
                channel_id = self.get_channel_id(channel)
                if not channel_id:
                    continue
            videos = self.search_videos_in_channel(channel_id, query, videos_per_channel)
            all_videos.extend(videos)
        return all_videos

def youtube_recommend(state: State, config: RunnableConfig):
    api_key = None
    try:
        api_key = st.secrets.REST.YOUTUBE_API_KEY
    except Exception:
        api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set")
    recommender = YouTubeVideoRecommender(api_key)
    favorite_channels = [
        "ZEE Business","Economic Times","Times Now","Times Now Business",
        "Times Now News","Times Now Politics","Times Now Sports","Times Now Science",
        "Times Now Technology","Pranjal Kamra","Yadnya Investment Academy",
        "CA Rachana Phadke Ranade","Invest Aaj For Kal","Market Gurukul",
        "Warikoo","Asset Yogi","Trading Chanakya","Trade Brains","B Wealthy",
        "Capital Pritika","The Urban Fight","Kritika Yadav","Gurleen Kaur Tikku"
    ]
    query = state.get("research_topic", "")
    recommendations = recommender.recommend_videos(query, favorite_channels, videos_per_channel=1)
    if not recommendations:
        summary = f"No matching videos found for query: {query}"
    else:
        summary = f"## YouTube Video Recommendations for '{query}'\n\n"
        for i, video in enumerate(recommendations, 1):
            summary += f"### {i}. {video.get('title')}\n"
            summary += f"- Channel: {video.get('channel')}\n"
            summary += f"- URL: {video.get('url')}\n"
            summary += f"- Published: {video.get('published_at')}\n\n"
    return {"running_summary": summary, "messages": [HumanMessage(content=summary)]}

# ---------------------------
# Self evaluation & evaluation_decision (kept)
# ---------------------------
def self_evaluate(input_text):
    parts = input_text.split("|||")
    query = parts[0]
    response = parts[1] if len(parts) > 1 else ""
    sources = parts[2] if len(parts) > 2 else ""

    evaluation_prompt = f"""
    Evaluate the following response to the query:

    QUERY: {query}
    RESPONSE: {response}
    SOURCES: {sources}

    Assess based on:
    1. Factual accuracy
    2. Completeness
    3. Relevance
    4. Hallucination
    Return a confidence score from 0-10 and a brief explanation.
    """
    eval_resp = llm.invoke(evaluation_prompt)
    return eval_resp.content

def evaluate_response(state: State, config: RunnableConfig):
    query = state.get("research_topic", "")
    response = state.get("running_summary", "")
    sources = "\n".join(state.get("sources_gathered", [])) or "No sources available"
    input_text = f"{query}|||{response}|||{sources}"
    evaluation = self_evaluate(input_text)
    final_summary = response
    return {"running_summary": final_summary, "messages": [HumanMessage(content=final_summary)]}

def evaluation_decision(state: State, config: RunnableConfig):
    final_text = state.get("running_summary", "")
    prompt = f"""
    The final output and self-evaluation are as follows:
    {final_text}

    Based on the above, do you think additional insights should be added?
    If yes, return a JSON object with the key "next_route" set to one of:
      - "call_llm", "web_research", "answer_normal_query"
    If no additional insights are needed, return "done".
    """
    result = llm.invoke(prompt)
    output_text = result.content.strip()
    try:
        decision = json.loads(output_text)
        next_route = decision.get("next_route", "done")
    except Exception:
        next_route = "done"
    state["next_route"] = next_route
    return {"next_route": next_route}

# ---------------------------
# Context processing (unchanged)
# ---------------------------
def process_with_context(state: State):
    messages = state.get("messages", [])
    original_messages = state.get("original_messages", [])
    if len(messages) <= 1:
        return {"messages": messages, "original_messages": original_messages, "research_topic": state.get("research_topic", "")}
    current_query = messages[-1].content
    if not original_messages or original_messages[-1].content != current_query:
        original_messages.append(HumanMessage(content=current_query))
    context_messages = messages[:-1]
    context_str = "\n".join([f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}" for i, msg in enumerate(context_messages[-6:])])
    prompt = f"""
    Based on the previous conversation context and the user's current query, generate an enhanced version of the query that incorporates relevant context where appropriate.

    Previous conversation:
    {context_str}

    Current query: {current_query}

    Enhanced query:
    """
    try:
        enhanced_query = llm.invoke(prompt).content.strip()
        updated_messages = messages[:-1] + [HumanMessage(content=enhanced_query)]
        return {"messages": updated_messages, "original_messages": original_messages, "research_topic": enhanced_query}
    except Exception:
        return {"messages": messages, "original_messages": original_messages, "research_topic": state.get("research_topic", "")}

# ---------------------------
# Router construction (keeps your graph)
# ---------------------------
def call_route_first_step(state: State):
    image_processed = state.get("image_processed", False)

    if state.get("image") and len(state["image"]) > 0 and not image_processed:
        return {
            "route": "Image_Analysis",
            "original_messages": state["original_messages"]
        }

    router_response = llm.with_structured_output(Route_First_Step).invoke(
        state["research_topic"]
    )

    print(f"Routing result: {router_response.step}")

    return {
        "route": router_response.step,
        "original_messages": state["original_messages"]
    }

def get_route(state: State) -> str:
    return state["route"]

def update_router():
    final_router = StateGraph(State)
    final_router.add_node("route_first_step", call_route_first_step)
    final_router.add_node("generate_query", generate_query)
    final_router.add_node("web_research", web_research)
    final_router.add_node("summarize_sources", summarize_sources)
    final_router.add_node("reflect_on_summary", reflect_on_summary)
    final_router.add_node("finalize_summary", finalize_summary)
    final_router.add_node('call_llm', call_llm)
    final_router.add_node('take_action', take_action)
    final_router.add_node('format_financial_analysis', format_financial_analysis)
    final_router.add_node('answer_normal_query', answer_normal_query)
    final_router.add_node('youtube_recommend', youtube_recommend)
    final_router.add_node("self_evaluate_final", evaluate_response)
    final_router.add_node("evaluation_decision", evaluation_decision)
    final_router.add_node("process_with_context", process_with_context)
    final_router.add_node("image_analysis", call_gemma3 if 'call_gemma3' in globals() else (lambda s: s))
    final_router.add_node("parse_query", parse_query)
    final_router.add_node("generate_plot", generate_plot)
    final_router.add_node("format_response", format_response)

    final_router.add_edge(START, "process_with_context")
    final_router.add_edge("process_with_context", "route_first_step")

    final_router.add_conditional_edges("route_first_step", get_route, {
        'Image_Analysis': 'image_analysis',
        'Web_query': 'generate_query',
        'Normal_query': 'answer_normal_query',
        'Financial_Analysis': 'call_llm',
        'YouTube_Recommender': 'youtube_recommend',
        'Plot_Graph': 'parse_query'
    })

    final_router.add_edge("parse_query", "generate_plot")
    final_router.add_edge("generate_plot", "format_response")
    final_router.add_edge("image_analysis", END)

    final_router.add_edge("answer_normal_query", 'self_evaluate_final')
    final_router.add_edge("format_response", 'self_evaluate_final')

    final_router.add_conditional_edges(
        "call_llm",
        exists_action,
        {True: "take_action", False: "format_financial_analysis"}
    )
    final_router.add_edge("take_action", "format_financial_analysis")
    final_router.add_edge("format_financial_analysis", END)

    final_router.add_edge("generate_query", "web_research")
    final_router.add_edge("web_research", "summarize_sources")
    final_router.add_edge("summarize_sources", "reflect_on_summary")
    final_router.add_conditional_edges("reflect_on_summary", route_research)
    final_router.add_edge("finalize_summary", 'self_evaluate_final')
    final_router.add_edge("self_evaluate_final", 'evaluation_decision')

    final_router.add_conditional_edges("evaluation_decision", lambda x: x.get("next_route", "done"), {
        'done': END,
        'call_llm': 'call_llm',
        'web_research': 'web_research',
        'answer_normal_query': 'answer_normal_query',
        'YouTube_Recommender': 'youtube_recommend'
    })
    final_router.add_edge("youtube_recommend", END)
    return final_router.compile()

# ---------------------------
# FinancialChatBot class (keeps behaviour)
# ---------------------------
class FinancialChatBot:
    def __init__(self, language='english'):
        self.conversation_history = []
        self.model = update_router()
        self.context_messages = []
        self.language = language

    # -----------------------------
    # Message formatting helpers
    # -----------------------------
    def _format_bot_message(self, content: str) -> str:
        return f"🤖 Assistant: {content}"

    def _format_user_message(self, content: str) -> str:
        return f"👤 User: {content}"

    # -----------------------------
    # Context memory
    # -----------------------------
    def _update_context(self, user_input: str, bot_response: str):
        self.context_messages.append(HumanMessage(content=user_input))
        self.context_messages.append(AIMessage(content=bot_response))
        if len(self.context_messages) > 10:
            self.context_messages = self.context_messages[-10:]

    # -----------------------------
    # Contextual enhancer
    # -----------------------------
    def _process_with_context(self, user_input: str):
        if not self.context_messages:
            return user_input

        context_system_prompt = """
        You are a financial assistant analyzing a conversation history.
        Given the conversation history and a new user query, your task is to:
        1. Understand the context of the ongoing conversation
        2. Generate an enhanced version of the user's query that incorporates relevant context
        3. Return ONLY the enhanced query without any explanations
        """

        context_prompt = "Conversation history:\n"
        for msg in self.context_messages[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            context_prompt += f"{role}: {msg.content}\n\n"

        context_prompt += f"New user query: {user_input}\n\nGenerate an enhanced query:"

        try:
            messages = [
                SystemMessage(content=context_system_prompt),
                HumanMessage(content=context_prompt),
            ]
            enhanced_query = llm.invoke(messages).content.strip()
            return enhanced_query
        except Exception:
            return user_input

    # -----------------------------
    # Main chat handler
    # -----------------------------
    def chat(self, user_input: str, image_path: str = None) -> dict:

        # Log user message
        self.conversation_history.append(self._format_user_message(user_input))

        # Apply context enhancer (if no image)
        contextualized_input = (
            self._process_with_context(user_input)
            if self.context_messages and not image_path
            else user_input
        )

        # Prepare state
        image_list = [image_path] if image_path else []
        initial_state = create_initial_state(contextualized_input, image_list)

        if self.context_messages:
            initial_state["messages"] = self.context_messages + [
                HumanMessage(content=contextualized_input)
            ]

        # -----------------------------
        # Model execution
        # -----------------------------
        try:
            response = self.model.invoke(initial_state)

            # Safe extraction of output
            if isinstance(response, dict):
                text_response = response.get("running_summary", "") or ""
                plot_json = response.get("plot_json", None)

                if not text_response and response.get("messages"):
                    text_response = response["messages"][-1].content

            elif hasattr(response, "message"):
                text_response = response.message.content
                plot_json = None

            elif isinstance(response, list) and len(response) > 0 and hasattr(response[0], "message"):
                text_response = response[0].message.content
                plot_json = None

            else:
                text_response = str(response)
                plot_json = None

        except Exception as e:
            text_response = f"{str(e)}"
            plot_json = None

        # -----------------------------
        # Translation (Sutra)
        # -----------------------------
        try:
            sutra_key = st.secrets.REST.SUTRA_API_KEY
        except Exception:
            sutra_key = os.environ.get("SUTRA_API_KEY")

        # Update memory
        self._update_context(user_input, text_response)

        # Log bot message
        self.conversation_history.append(self._format_bot_message(text_response))

        # Translate if required
        if self.language != "english" and sutra_key:
            client = OpenAI(base_url="https://api.two.ai/v2", api_key=sutra_key)
            stream = client.chat.completions.create(
                model="sutra-v2",
                messages=[
                    {
                        "role": "user",
                        "content": f"Translate this text in {self.language}: {text_response}",
                    }
                ],
                max_tokens=1024,
                temperature=0,
                stream=False,
            )
            choice = stream.choices[0]

            # Works for object (most cases)
            if hasattr(choice.message, "content"):
                translated_text = choice.message.content

            # Works for dict fallback
            elif isinstance(choice.message, dict):
                translated_text = choice.message.get("content", "")

            else:
                translated_text = str(choice)

            return {"text": translated_text, "plot": plot_json}

        return {"text": text_response, "plot": plot_json}

    # -----------------------------
    # History management
    # -----------------------------
    def get_conversation_history(self) -> str:
        return "\n\n".join(self.conversation_history)

    def clear_history(self):
        self.conversation_history = []
        self.context_messages = []


# ---------------------------
# Optional command-line main (keeps behavior)
# ---------------------------
def main():
    chatbot = FinancialChatBot()
    sutra_key = os.environ.get("SUTRA_API_KEY")
    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() == 'quit':
            print("\nGoodbye! Thank you for using the Financial Assistant.")
            break
        image_path = None
        if user_input.startswith('image:'):
            image_path = user_input[6:].strip()
            user_input = "What do you see in this image?"
        response = chatbot.chat(user_input, image_path)
        print("\n🤖 Assistant:\n", response.get('text'))

if __name__ == "__main__":
    main()