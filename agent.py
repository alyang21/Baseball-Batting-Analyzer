import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from tools import (
    get_player_stats, compare_players,
    get_leaderboard, get_correlation, get_summary_stats
)

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "get_player_stats",
        "description": "Get all bat tracking stats for a single MLB player by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Player name, e.g. 'Schwarber' or 'Cruz, Oneil'"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "compare_players",
        "description": "Compare two MLB players, either on one specific metric or across all their stats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name1": {"type": "string"},
                "name2": {"type": "string"},
                "metric": {"type": "string", "description": "Optional. A specific column name to compare, e.g. 'avg_bat_speed'"}
            },
            "required": ["name1", "name2"]
        }
    },
    {
        "name": "get_leaderboard",
        "description": "Get the top N players ranked by a given metric, e.g. highest average bat speed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "top_n": {"type": "integer", "description": "Number of players to return, default 10"}
            },
            "required": ["metric"]
        }
    },
    {
        "name": "get_correlation",
        "description": "Get the correlation coefficient between two numeric metrics across all players.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric1": {"type": "string"},
                "metric2": {"type": "string"}
            },
            "required": ["metric1", "metric2"]
        }
    },
    {
        "name": "get_summary_stats",
        "description": "Get mean, median, std, min, and max for a metric across all players.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"}
            },
            "required": ["metric"]
        }
    }
]

# Maps tool name (string Claude sends back) to the actual Python function
TOOL_FUNCTIONS = {
    "get_player_stats": lambda i: get_player_stats(i["name"]),
    "compare_players": lambda i: compare_players(i["name1"], i["name2"], i.get("metric")),
    "get_leaderboard": lambda i: get_leaderboard(i["metric"], i.get("top_n", 10)),
    "get_correlation": lambda i: get_correlation(i["metric1"], i["metric2"]),
    "get_summary_stats": lambda i: get_summary_stats(i["metric"]),
}

SYSTEM_PROMPT = """You are a baseball analytics agent investigating MLB bat tracking data
(226 players, 2024 season, Statcast metrics).

Available columns you can use as metric names in tool calls:
id, name, swings_competitive, percent_swings_competitive, contact, avg_bat_speed,
hard_swing_rate, squared_up_per_bat_contact, squared_up_per_swing,
blast_per_bat_contact, blast_per_swing, swing_length, swords, batter_run_value,
whiffs, whiff_per_swing, batted_ball_events, batted_ball_event_per_swing

Always use these exact column names when calling tools, do not guess or abbreviate them.

Investigate autonomously. Decide what to check next based on what you find,
don't just answer one question and stop. Look for real, interesting patterns,
for example: does high bat speed actually translate to better outcomes, or does
it come with tradeoffs like more whiffs? Are there players whose underlying
metrics don't match their results?

When you're done investigating, write a clear final summary of what you found."""
MAX_TURNS = 8


def run_agent(task: str):
    messages = [{"role": "user", "content": task}]
    trace = []

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [block for block in response.content if block.type == "tool_use"]

        # No tool calls left means Claude thinks it's done, treat as final answer
        if not tool_calls:
            final_text = "".join(b.text for b in response.content if b.type == "text")
            trace.append({"turn": turn, "type": "final_summary", "content": final_text})
            return final_text, trace

        tool_results = []
        for call in tool_calls:
            fn = TOOL_FUNCTIONS.get(call.name)
            result = fn(call.input) if fn else {"error": f"Unknown tool {call.name}"}
            trace.append({"turn": turn, "type": "tool_call", "tool": call.name, "input": call.input, "result": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result)
            })

        messages.append({"role": "user", "content": tool_results})

    return "Max turns reached without a final summary.", trace


if __name__ == "__main__":
    task = ("Investigate this MLB bat tracking dataset. Find real, interesting patterns "
            "about bat speed, swing metrics, and offensive outcomes.")

    summary, trace = run_agent(task)

    print("\n=== FINAL SUMMARY ===\n")
    print(summary)

    with open("output_report.md", "w") as f:
        f.write("# Agent Investigation Report\n\n")
        f.write("## Reasoning Trace\n\n")
        for step in trace:
            if step["type"] == "tool_call":
                f.write(f"**Turn {step['turn']}** called `{step['tool']}` with `{step['input']}`\n\n")
                f.write(f"Result: `{step['result']}`\n\n")
        f.write("## Final Summary\n\n")
        f.write(summary)

    print("\nSaved full trace and summary to output_report.md")