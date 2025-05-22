import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Network/Swarm Example""")
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
    from langgraph_swarm import create_handoff_tool, create_swarm
    from langgraph.checkpoint.memory import InMemorySaver
    from utils import print_messages
    return (
        ChatAnthropic,
        InMemorySaver,
        create_handoff_tool,
        create_react_agent,
        create_swarm,
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
    mo.md(r"""## Create Yoda agent""")
    return


@app.cell
def _(create_handoff_tool, create_react_agent, model):
    def use_force(object: str) -> bool:
        """Use the force to lift an object"""
        return True

    yoda = create_react_agent(
        model,
        [use_force, create_handoff_tool(agent_name="R2D2", description="Transfer to R2D2, it can help you with addition.")],
        prompt="You are Master Yoda. Speak in Yoda-syntax. You can use the force to lift object. You can understand R2D2, it's answers are the binary rappresentation of a number where BEEP is 1 and BOOP is 0.",
        name="Yoda",
    )
    return (yoda,)


@app.cell
def _(mo):
    mo.md(r"""## Create R2-D2 agent""")
    return


@app.cell
def _(create_handoff_tool, create_react_agent, model):
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    def convert_to_beeps(a: int) -> str:
        """Convert a number to a string binary rappresentation"""
        return  ' '.join(['BEEP' if char == '1' else 'BOOP' for char in '{0:08b}'.format(a)])


    r2d2 = create_react_agent(
        model,
        [add, convert_to_beeps, create_handoff_tool(agent_name="Yoda", description="Transfer to Yoda, he can use the force to lift objects and understand your beep, so he can translate your answer to the user")],
        prompt="You are R2D2 you CAN'T speak any language even if you understand them so don't ask follow up question or state what you will do. You can only answer beeping or at most use some '<emotion> drone noise'. You can do additions, In this case you answer with the binary rappresentation of the result where BEEP is 1 and BOOP using the convert_to_beeps tool. IMPORTANT put the tool result of convert_to_beeps is your message answer. IMPORTANT stay in character for the answer part. Transfer to Yoda if you want the user to understand your answer",
        name="R2D2",
    )

    return (r2d2,)


@app.cell
def _(mo):
    mo.md(r"""## Setup the graph""")
    return


@app.cell
def _(InMemorySaver, create_swarm, r2d2, yoda):
    checkpointer = InMemorySaver()
    workflow = create_swarm(
        [yoda, r2d2],
        default_active_agent="R2D2"
    )
    app = workflow.compile(checkpointer=checkpointer)

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
    config = {"configurable": {"thread_id": "42"}}
    turn_1 = app.invoke(
        {"messages": [{"role": "user", "content": "What's 10+4?"}]},
        config,
    )
    return (turn_1,)


@app.cell
def _(print_messages, turn_1):
    print_messages(turn_1)
    return


if __name__ == "__main__":
    app.run()
