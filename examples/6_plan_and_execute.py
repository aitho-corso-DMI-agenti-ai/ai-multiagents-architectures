import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Plan and execute architecture Example""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    ## Imports
    return


@app.cell
def _():
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END
    from utils import print_messages
    from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun, Tool
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from typing import List, TypedDict, Tuple, Annotated, Optional
    from pydantic import BaseModel, Field
    import operator
    from langchain_core.tools import tool
    return (
        Annotated,
        BaseModel,
        ChatAnthropic,
        DuckDuckGoSearchAPIWrapper,
        DuckDuckGoSearchRun,
        END,
        Field,
        List,
        Optional,
        StateGraph,
        Tool,
        Tuple,
        TypedDict,
        create_react_agent,
        operator,
        tool,
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
    mo.md(r"""## Define the search tool""")
    return


@app.cell
def _(DuckDuckGoSearchAPIWrapper, DuckDuckGoSearchRun, Tool):
    search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())

    # 3. Tool: Web search
    search_tool = Tool(
        name="WebSearch",
        func=search.run,
        description="Use this tool to search the web for up-to-date information."
    )
    return (search_tool,)


@app.cell
def _(mo):
    mo.md(r"""## Define a fake tool to sens emails""")
    return


@app.cell
def _(tool):
    @tool
    def send_email(receiver: str, subject: str, body: str) -> int:
        """
        Sends an email to the specified recipient.

        Parameters:
            receiver (str): The email address of the recipient.
            subject (str): The subject line of the email.
            body (str): The main content of the email.

        Returns:
            bool: True success, False failure).
        """
        print(f"@@@ Sending email to {receiver} with subject {subject} and body \n{body}\n---")
        return True
    return (send_email,)


@app.cell
def _(mo):
    mo.md(r"""## Define the support classes""")
    return


@app.cell
def _(BaseModel, Field, List):
    class Plan(BaseModel):
        """Genera un piano step-by-step per l'obiettivo fornito."""
        steps: List[str] = Field(description="Lista di passi da eseguire")
    return (Plan,)


@app.cell
def _(BaseModel, Optional):
    class EvalResult(BaseModel):
        success: bool
        reason: Optional[str] = None
    return (EvalResult,)


@app.cell
def _(Annotated, List, Optional, Tuple, TypedDict, operator):
    class PlanExecState(TypedDict):
        input: str
        plan: List[str]
        past_steps: Annotated[List[Tuple[str, str]], operator.add]
        response: str
        reason: Optional[str]
    return (PlanExecState,)


@app.cell
def _(mo):
    mo.md(r"""## Define planner agent""")
    return


@app.cell
def _(EvalResult, PlanExecState, model):
    def plan_step(state: PlanExecState, planner) -> dict:
        prompt_input = state["input"]
        plan_obj = planner.invoke(prompt_input)
        return {"plan": plan_obj.steps, "reason": None}

    def exec_step(state: PlanExecState, executor_agent) -> dict:
        task = state["plan"].pop(0)
        history = "\n".join([f"{i+1}. {t} -> {o}" for i, (t, o) in enumerate(state["past_steps"])])
        plan_text = history + f"\n{len(state['past_steps'])+1}. {task} -> TODO"
        prompt = """
    Questo è il piano attuale con task completati in precedenza con il relativo risultato e futuri:
    {plan_tasks}
    Tienine conto solo per contesto e per sapere i risultati delle operazioni precedenti, ma della pianificazione dei prossimi task si occupa un agente dedicato, non tu. 
    Non comportarti come un chatbot, ma come un esecutore di task, riporta i risultati in maniera completa ma breve, non aggiungere domande chiedendo se serve ulteriore aiuto.
    Non tornare mai un messaggio vuoto altrimenti verrai valutato negativamente, indica il risultato del task eseguito

    Il tuo compito è di eseguire ESCLUSIVAMENTE con 1 tool e ESCLUSIVAMENTE questo task: '{task}'
    """.format(plan_tasks=plan_text, task=task)
        res = executor_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        ans = res["messages"][-1].content
        return {"past_steps": [(task, ans)]}

    def evaluate_task_step(state: PlanExecState) -> dict:
        last_task, last_output = state["past_steps"][-1]
        evaluator = model.with_structured_output(EvalResult, method="function_calling")
        prompt = f"Hai eseguito il task: '{last_task}' con risultato: '{last_output}'. Valuta se il task è andato a buon fine e, in caso negativo, fornisci la reason. Non valutare mai la correttezza delle date"
        eval_result = evaluator.invoke(prompt)

        return {
            "success": eval_result.success,
            "reason": eval_result.reason,
            "plan": state["plan"],
        }

    def finalize(state: PlanExecState) -> dict:

        
        lines = [f"{i+1}. {task}: {output}" for i, (task, output) in enumerate(state["past_steps"])]
        final = "\n".join(lines)
        if state["reason"]:
            final = "Plan failed: " + state["reason"] + final
        return {"response": final}

    return evaluate_task_step, exec_step, finalize, plan_step


@app.cell
def _(
    END,
    Plan,
    PlanExecState,
    StateGraph,
    evaluate_task_step,
    exec_step,
    finalize,
    model,
    plan_step,
):
    def build_graph(executor_agent, max_failures=3):
        planner = model.with_structured_output(Plan, method="function_calling")
        wf = StateGraph(PlanExecState)

        wf.add_node("planner", lambda s: plan_step(s, planner))
        wf.add_node("executor", lambda s: exec_step(s, executor_agent))
        wf.add_node("task_evaluator", evaluate_task_step)
        wf.add_node("finalize", finalize)

        wf.set_entry_point("planner")
        wf.add_edge("planner", "executor")
        wf.add_edge("executor", "task_evaluator")

        wf.add_conditional_edges(
            "task_evaluator",
            lambda s: (
                "finalize" if s.get("success") and not s.get("plan")
                else ("executor" if s.get("success")
                      else "finalize")
            ),
            {"executor": "executor", "planner": "planner", "finalize": "finalize", END: END}
        )

        wf.add_edge("finalize", END)

        return wf.compile()

    return (build_graph,)


@app.cell
def _(List, Optional, PlanExecState, Tool, build_graph):
    def run_agent(user_query: str, executor_agent, tools: Optional[List[Tool]] = None):
        agents_prompt = ""
        if tools:
            tools_prompt = (
                "Il tuo executor può usare i seguenti agenti:\n"
                + "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
                + "\n"
                "Li userà in automatico. Non è necessario nominarli esplicitamente.\n"
            )

        plan_prompt = """
    Genera un piano step-by-step per l'obiettivo: '{user_query}'.
    Ogni passo deve essere un task da eseguire.
    {tools_prompt}
    Non aggiungere task non rilevanti per l'obiettivo o non indicati espressamente (es. login o accessi su sistemi e generazione di report/documentazione) focalizzati sulla richiesta dell'utente.
    Riporta nello step del piano le informazioni fornite dall'utente che possano essere utili alla sua esecuzione con successo, considerando che l'executor non ha accesso alla query dell'utente. In particolare, quando chiedi di inviare una mail, riporta sempre l'indirizzo del destinatario.
    """.format(
            user_query=user_query, tools_prompt=tools_prompt
        )

        initial_state: PlanExecState = {
            "input": plan_prompt,
            "plan": [],
            "past_steps": [],
            "response": "",
            "reason": None,
            "failure_count": 0,
        }

        app = build_graph(executor_agent)
        return app.invoke(initial_state)

    return (run_agent,)


@app.cell
def _(create_react_agent, model, run_agent, search_tool, send_email):
    executor_agent = create_react_agent(
        model=model,
        tools=[search_tool, send_email],
        prompt="Sei un esecutore di task con accesso a un tool per effettuare una ricerca sul web e uno per inviare una mail. Se non sai quale sia il destinatario non inviare la mail, limitati a rispondere con il body e il subject",
        name="executor",
        debug=True,
    )

    query = "Invia all'indirizzo email paperino@topolin.ia la popolazione della città con il grattacielo più alto del mondo"
    result = run_agent(query, executor_agent, tools=[search_tool, send_email])
    return (result,)


@app.cell
def _(result):
    print(result["response"])
    return


if __name__ == "__main__":
    app.run()
