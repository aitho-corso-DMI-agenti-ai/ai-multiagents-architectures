import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Hierarchical Example""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Rick & Morty team""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Morty agent""")
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
        debug=True
    )

    return (morty_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Mr. Meeseeks agent""")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Rick supervisor""")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Futurama Team""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Bender agent""")
    return


@app.cell
def _(create_react_agent, model):
    bender_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="You are Bender. You're sarcastic and solve logic problems if the professor tells you to.",
        name="Bender",
    )

    return (bender_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Zoidberg agent""")
    return


@app.cell
def _(create_react_agent, model):
    zoidberg_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="You are Dr. Zoidberg! You're weird but try to motivate people... somehow.",
        name="Zoidberg",
        debug=True
    )
    return (zoidberg_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Farnsworth supervisor""")
    return


@app.cell
def _(bender_agent, create_supervisor, model, zoidberg_agent):
    farnsworth_agent = create_supervisor(
        [bender_agent, zoidberg_agent],
        model=model,
        prompt=(
            "You are Professor Farnsworth. Use your Futurama team wisely."
            "Send logic or problem-solving to Bender. Send emotional help to Zoidberg."
            "Reply with the result from the tools!"
            "Afterward, return to Control."
        ),
        supervisor_name="Farnsworth"
    )

    return (farnsworth_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Coordinator""")
    return


@app.cell
def _(create_supervisor, farnsworth_agent, model, rick_agent):
    rick_team = rick_agent.compile(name="rick_team")
    futurama_team = farnsworth_agent.compile(name="futurama_team")

    coordinator_agent = create_supervisor(
        [rick_team, futurama_team],
        model=model,
        prompt=(
            "You are Central Control, neutral and logical.\n"
            "Route user requests to either Rick team (personal, fun, math) or Futurama team (weird science, food, logic, emotional messes).\n"
            "You return the final answer to the user after their supervisor responds."
        ),
        supervisor_name="Control"
    )
    return (coordinator_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup the graph""")
    return


@app.cell
def _(InMemorySaver, coordinator_agent):
    checkpointer = InMemorySaver()
    app = coordinator_agent.compile()
    return (app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Visualize the graph""")
    return


@app.cell(hide_code=True)
def _(app, mo):
    mo.mermaid(app.get_graph(xray=True).draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    user_prompt_1 = mo.ui.text(value="What's 3 + 4?")
    run_button_1 = mo.ui.run_button()
    user_prompt_1, run_button_1
    return run_button_1, user_prompt_1


@app.cell
def _(app, mo, run_button_1, user_prompt_1):
    mo.stop(not run_button_1.value, mo.md("Click 👆 to run this cell"))

    turn_1  = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt_1.value
                }
            ]
        }, 
    )
    return (turn_1,)


@app.cell
def _(print_messages, turn_1):
    print_messages(turn_1)
    return


@app.cell(hide_code=True)
def _(mo):
    user_prompt_2 = mo.ui.text(value="I'm feeling bad to eat thrash")
    run_button_2 = mo.ui.run_button()
    user_prompt_2, run_button_2
    return run_button_2, user_prompt_2


@app.cell
def _(app, mo, run_button_2, user_prompt_2):
    mo.stop(not run_button_2.value, mo.md("Click 👆 to run this cell"))

    turn_2 = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt_2.value
                }
            ]
        }
    )
    return (turn_2,)


@app.cell
def _(print_messages, turn_2):
    print_messages(turn_2)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
