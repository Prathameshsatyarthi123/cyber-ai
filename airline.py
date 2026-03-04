from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import sqlite3
import traceback
import json

load_dotenv(override=True)

MODEL = "llama-3.1-8b-instant"
DB = "prices.db"

initial_ticket_prices = {"london": 799, "paris": 899, "tokyo": 1400, "sydney": 2999}

with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)")
    for city, price in initial_ticket_prices.items():
        cursor.execute(f"INSERT OR IGNORE INTO prices (city, price) VALUES ('{city}', {price})")
    conn.commit()

client = OpenAI(
    api_key=__import__('os').environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

instructions = "You are a helpful assistant for an Airline called FlightAI. "
instructions += "Use your tools to get ticket prices and calculate discounts. Trips to London have a 10% discount on the price. "
instructions += "Always be accurate. If you don't know the answer, say so."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "Get the price of a ticket to a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get the price of a ticket to"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a numeric expression - use this for calculations about prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "The expression to evaluate"}
                },
                "required": ["expr"]
            }
        }
    }
]


def get_ticket_price(city: str) -> str:
    print(f"TOOL CALLED: Getting price for {city}", flush=True)
    try:
        with sqlite3.connect(DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT price FROM prices WHERE city = ?", (city.lower(),))
            result = cursor.fetchone()
            return f"${result[0]}" if result else "Not found"
    except Exception:
        return f"Error: {traceback.format_exc()}"


def calculate(expr: str) -> str:
    print(f"TOOL CALLED: Calculating {expr}", flush=True)
    return str(eval(expr))


def handle_tool_call(tool_name: str, tool_args: dict) -> str:
    if tool_name == "get_ticket_price":
        return get_ticket_price(tool_args["city"])
    elif tool_name == "calculate":
        return calculate(tool_args["expr"])
    return "Unknown tool"


async def chat(message, history):
    messages = [{"role": "system", "content": instructions}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    # Agentic loop - keep running until no more tool calls
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # Add assistant message with tool calls
            messages.append(choice.message)

            # Execute each tool and add results
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                result = handle_tool_call(tool_name, tool_args)
                print(f"Tool result: {result}", flush=True)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # No more tool calls, return final response
            return choice.message.content


gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
