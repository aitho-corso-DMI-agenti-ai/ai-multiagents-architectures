import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Simple Graph LLM Example
    This is a simple example of a LangGraph that demonstrates how to create a graph with conditional edges and how to invoke it with different inputs using LLMs.

    The graph consists of two nodes: "_greeting_" and "_emoji_". 

    The "_greeting_" node generates a greeting message based on the input name, and the "_emoji_" node appends an emoji to the text.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define the imports""")
    return


@app.cell
def _():
    from langgraph.graph import Graph, START, END
    from langchain_anthropic import ChatAnthropic
    import json
    return ChatAnthropic, END, Graph, START, json


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
    mo.md(r"""## Define the greeting node""")
    return


@app.cell
def _(json, model):
    # Function to create the greeting
    # Inputs is the input of the compiled graph invoke, because it is the first node
    def greeting_node(inputs):
        name_input = inputs.get("name_input")

        prompt = ('Rispondi usando questa struttura di JSON: { "text": "Messaggio generato", "add_emoji": "boolean"}'
                f"Genera un messaggio di saluto per lo studente {name_input} che sta frequentando il corso di Aitho sugli agenti e se si chiama Gabriele, setta il flag 'add_emoji' a true.")
        messages = [
            ("human", prompt),
        ]
        response = model.invoke(messages)
        return json.loads(response.content)
    return (greeting_node,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define the emoji node""")
    return


@app.cell
def _(model):
    # Function to add emoji to the text
    # Inputs is the *output* of the previous node
    def emoji_node(inputs):
        text = inputs.get("text", "")
        print("Original message: ", text)
        prompt = (f"Aggiungi una emoji super mega swag con rizz al seguente testo '{text}'."
                  "IMPORTANTE: rispondi solo con il testo e l'emoji, non aggiungere altro!")
        messages = [
            ("human", prompt),
        ]
        response = model.invoke(messages)
        return {"text": response.content}
    return (emoji_node,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define the conditional edge to choose the next node to execute
    If the input contains the attribute "add_emoji" sets to True we execute the _emoji_ node next, otherwise we terminate the graph execution in the _END_ node
    """
    )
    return


@app.function
# Function to determine the next node based on the input
def next_node_after_greeting(inputs):
    # If the input contains add_emoji to True, go to the "emoji" node
    if inputs.get("add_emoji", False):
        return "if add_emoji"
    # Otherwise, go to the END node
    # END is a special node that indicates the end of the graph. If it's missing, the graph will not return anything.
    return "else"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Let's create the graph""")
    return


@app.cell
def _(END, Graph, START, emoji_node, greeting_node):
    graph = Graph()

    graph.add_node("greeting", greeting_node)
    graph.add_node("emoji", emoji_node)

    graph.add_edge(START, "greeting")
    # or graph.set_entry_point("greeting")
    graph.add_edge("emoji", END) 
    # or graph.set_finish_point("emoji")

    # Add a conditional edge from the "greeting" node to the node name returned by the function an mapped by path_map
    graph.add_conditional_edges("greeting", next_node_after_greeting, path_map={
        "if add_emoji": "emoji",
        "else": END
    })
    return (graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compiling the graph
    Before we can execute a graph, we need to compile it
    """
    )
    return


@app.cell
def _(graph):
    compiled_graph = graph.compile()
    return (compiled_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Visualize the graph""")
    return


@app.cell(hide_code=True)
def _(compiled_graph, mo):
    mo.mermaid(compiled_graph.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Let's test it""")
    return


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button()
    run_button
    return (run_button,)


@app.cell
def _(compiled_graph, mo, run_button):
    mo.stop(not run_button.value, mo.md("Click 👆 to run this cell"))
    tests = ["Mario", "Gabriele", "Pippo"]
    for user in tests:
        result = compiled_graph.invoke({"name_input": user})
        print(f"Input={user} -> {result.get('text')}")

    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
