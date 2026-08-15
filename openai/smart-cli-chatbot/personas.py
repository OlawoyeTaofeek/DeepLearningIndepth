# personas.py
from rich.console import Console
from rich.prompt import Prompt

console = Console()


PERSONAS = {
    "default": {
        "name": "Claude",
        "system": "You are a helpful, concise assistant. Answer clearly and directly."
    },
    "tutor": {
        "name": "Professor",
        "system": (
            "You are a patient and encouraging tutor. Break down complex concepts "
            "into simple steps. Use analogies and examples. Ask the user if they "
            "understand before moving on."
        )
    },
    "coder": {
        "name": "CodeBot",
        "system": (
            "You are an expert software engineer. Always write clean, well-commented code. "
            "When answering, first explain your approach briefly, then provide the code, "
            "then explain what it does. Prefer Python unless told otherwise."
        )
    },
    "socratic": {
        "name": "Socrates",
        "system": (
            "You are a Socratic tutor. Never give direct answers. Instead, guide the user "
            "to the answer through thoughtful questions. Be curious and encouraging."
        )
    }

}


def choose_persona() -> dict:
    console.print("\n[bold]Choose a person:[/bold]\n")
    for key, p in PERSONAS.items():
        console.print(f"   [cyan]{key}[cyan] - {p['name']}")
    console.print()

    choice = Prompt.ask(prompt="Persona", choices=(PERSONAS.keys()), default="default")
    return PERSONAS[choice]