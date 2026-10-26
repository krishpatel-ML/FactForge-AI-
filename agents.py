from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search ,scrape_url
from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.4,     
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0,
    rate_limiter=rate_limiter,
    max_retries=6,
    timeout=90,
)

def build_search_agent():
    return create_agent(
        model = llm,
        tools=[web_search]
    )

def build_reader_agent():
    return create_agent(
        model = llm,
        tools=[scrape_url]
    )

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])
writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
