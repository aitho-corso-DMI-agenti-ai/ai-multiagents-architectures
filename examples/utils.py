def print_messages(messages):
    for msg in messages["messages"]:
        tipo = type(msg).__name__  # es: 'HumanMessage', 'AIMessage'
        nome = getattr(msg, "name", "N/A")
        contenuto = msg.content
    
        # Se il contenuto è strutturato (lista di dict), estrai i testi
        if isinstance(contenuto, list):
            testo = "\n".join(block.get("text", "") for block in contenuto if block.get("type") == "text")
        else:
            testo = contenuto
    
        print(f"[Tipo: {tipo}] [Agente: {nome}]\nContenuto: {testo}\n{'-'*50}")