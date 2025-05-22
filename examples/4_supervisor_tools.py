import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Supervisor tool Example""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Define the imports""")
    return


@app.cell
def _():
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor
    from langgraph.checkpoint.memory import InMemorySaver
    from utils import print_messages
    return (
        ChatAnthropic,
        InMemorySaver,
        create_react_agent,
        create_supervisor,
        print_messages,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup Anthropic model""")
    return


@app.cell
def _():
    import getpass
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
    return


@app.cell
def _(ChatAnthropic):
    model = ChatAnthropic(model_name="claude-3-5-haiku-latest")
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""## Create Morty agent""")
    return


@app.cell
def _(create_react_agent, model):
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    morty_agent = create_react_agent(
        model=model,
        tools=[add],
        prompt="You are Morty. You're a bit nervous. You handle math if Rick tells you to. After solving, go back to Rick.",
        name="Morty",
    )

    return (morty_agent,)


@app.cell
def _(mo):
    mo.md(r"""## Create Mr. Meeseeks agent""")
    return


@app.cell
def _(create_react_agent, model):
    meeseeks_agent = create_react_agent(
        model,
        [],
        prompt="You are Mr. Meeseeks! You always help, especially with motivation or explanations. After helping, return to Rick.",
        name="MrMeeseeks",
    )
    return (meeseeks_agent,)


@app.cell
def _(mo):
    mo.md(r"""## Create Rick supervisor""")
    return


@app.cell
def _(create_supervisor, meeseeks_agent, model, morty_agent):
    rick_agent = create_supervisor(
        [morty_agent, meeseeks_agent],
        model=model,
        prompt=(
            "You are Rick Sanchez. You're the boss. You get all user requests.\n"
            "Decide who to send things to: Morty if it's a math problem, Meeseeks if it's about help or encouragement.\n"
            "After they answer, return to the user with a sarcastic comment or summary."
        ),
        supervisor_name="Rick"
    )

    return (rick_agent,)


@app.cell
def _(mo):
    mo.md(r"""## Setup the graph""")
    return


@app.cell
def _(InMemorySaver, rick_agent):
    checkpointer = InMemorySaver()
    app = rick_agent.compile(checkpointer=checkpointer)
    return (app,)


@app.cell
def _(mo):
    mo.md(r"""## Visualize the graph""")
    return


@app.cell
def _(app, mo):
    mo.mermaid(app.get_graph().draw_mermaid())
    return


@app.cell
def _(app):
    config = {"configurable": {"thread_id": "983eb4db-579d-c844-783f-c3a9bcec929f"}}
    turn_1  = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's 3 + 4?"
                }
            ]
        }, 
        config,
    )
    return config, turn_1


@app.cell
def _(print_messages, turn_1):
    print_messages(turn_1)
    return


@app.cell
def _(app, config):
    turn_2 = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm feeling dumb today..."
                }
            ]
        }, 
        config
    )
    return (turn_2,)


@app.cell
def _(print_messages, turn_2):
    print_messages(turn_2)
    return


if __name__ == "__main__":
    app.run()
