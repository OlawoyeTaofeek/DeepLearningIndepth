from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from typing import List

console = Console()

class TokenTracker:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.turn = 0


    def update(self, usage):
        # Handle both Anthropic and OpenAI usage objects
        if hasattr(usage, "input_tokens"):
            # Anthropic
            self.total_input += usage.input_tokens
            self.total_output += usage.output_tokens
        elif hasattr(usage, "prompt_tokens"):
            # OpenAI
            self.total_input += usage.prompt_tokens
            self.total_output += usage.completion_tokens
        else:
            pass  # unknown usage shape, skip silently
        self.turn += 1

    def display(self):
        table = Table(title="Token Usage Statistics", 
                      show_header=True, header_style="Bold cyan", padding=(0, 2))
        table.add_column("Metrics", style="dim", width=20)
        table.add_column("Tokens", style="magenta")

        table.add_row("Total Turns", str(self.turn))
        table.add_row("Input Tokens", f"{self.total_input:,}")
        table.add_row("Output Tokens", f"{self.total_output:,}")
        table.add_row("Total Tokens", f"{self.total_input + self.total_output:,}")
        table.add_row("Estimated Cost", f"${(self.total_input / 1_000_000) * 3.00 + (self.total_output / 1_000_000) * 15.00:.6f}")

        console.print(Panel(table, title="[dim]Session stats[/dim]", border_style="dim"))

def handle_command(cmd: str, tracker: TokenTracker, messages: List) -> bool:
    cmd = cmd.strip().lower()
    if cmd in ("/quit", "/q", "/exit"):
        tracker.display()
        console.print("\n[dim]Goodbye.[/dim]\n")

    elif cmd == "/stats":
        tracker.display()

    elif cmd == "/clear":
        messages.clear()
        console.print("\n[dim]Conversation cleared.[/dim]\n")
    
    elif cmd == "/help":
        help_text = (
            "[cyan]/stats[/cyan]   — show token usage\n"
            "[cyan]/clear[/cyan]   — clear conversation history\n"
            "[cyan]/history[/cyan] — show conversation so far\n"
            "[cyan]/help[/cyan]    — show this menu\n"
            "[cyan]/quit[/cyan]    — exit"
        )
        console.print(Panel(help_text, title="[dim]Commands[/dim]", border_style="dim"))

    elif cmd == "/history":
        if not messages:
            console.print("\n[dim]No messages yet[/dim]\n")
            return True  
        
        else:
            for i, m in enumerate(messages):
                role_color = "green" if m["role"] == "user" else "blue"
                content = m['content'][:300] + "..." if len(m['content']) > 300 else m['content']
                console.print(f"[{role_color}]{m['role'].upper()}[/{role_color}]: {content}\n")
    else:
        console.print(f"[dim]Unknown command: {cmd}. Type /help for options.[/dim]")

    return True