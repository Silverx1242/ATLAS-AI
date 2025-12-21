# commands/parser.py
"""
Implementation of the command parser for the system.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

from config import COMMAND_PREFIX, AVAILABLE_COMMANDS

logger = logging.getLogger(__name__)

class CommandParser:
    """Class to parse and validate user commands."""
    
    def __init__(self, command_prefix: str = COMMAND_PREFIX):
        """
        Initialize the command parser.
        
        Args:
            command_prefix: Prefix to identify commands
        """
        self.command_prefix = command_prefix
        self.available_commands = AVAILABLE_COMMANDS
    
    def is_command(self, text: str) -> bool:
        """
        Check if a text is a valid command.
        
        Args:
            text: Text to check
            
        Returns:
            True if it's a command, False otherwise
        """
        if not text.startswith(self.command_prefix):
            return False
        
        # Extract command name (without prefix)
        command_parts = text[len(self.command_prefix):].strip().split()
        if not command_parts:
            return False
        
        command_name = command_parts[0].lower()
        return command_name in self.available_commands
    
    def parse_command(self, text: str) -> Tuple[str, List[str], Dict[str, str]]:
        """
        Parse a command and extract its name, arguments and options.
        
        Args:
            text: Command text
            
        Returns:
            Tuple of (command_name, arguments, options)
        """
        if not self.is_command(text):
            raise ValueError(f"Not a valid command: {text}")
        
        # Remove prefix
        command_text = text[len(self.command_prefix):].strip()
        
        # Split respecting quotes for arguments with spaces
        parts = re.findall(r'(?:"[^"]*"|\S)+', command_text)
        
        # Extract command name
        command_name = parts[0].lower()
        
        # Process arguments and options
        args = []
        kwargs = {}
        
        for part in parts[1:]:
            # Check if it's an option with format --option=value
            if part.startswith("--") and "=" in part:
                opt_name, opt_value = part[2:].split("=", 1)
                # Remove quotes if present
                if opt_value.startswith('"') and opt_value.endswith('"'):
                    opt_value = opt_value[1:-1]
                kwargs[opt_name] = opt_value
            else:
                # It's a positional argument
                # Remove quotes if present
                if part.startswith('"') and part.endswith('"'):
                    part = part[1:-1]
                args.append(part)
        
        logger.debug(f"Command parsed: {command_name}, args={args}, kwargs={kwargs}")
        return command_name, args, kwargs
    
    def get_help_text(self, command_name: Optional[str] = None) -> str:
        """
        Generate help text for commands.
        
        Args:
            command_name: Specific command name (optional)
            
        Returns:
            Formatted help text
        """
        if command_name and command_name in self.available_commands:
            # Specific help for a command
            return self._get_specific_help(command_name)
        
        # General help
        help_text = [
            "Available commands:",
            f"{self.command_prefix}help [command] - Shows this help or specific help for a command"
        ]
        
        # Add basic descriptions for each command
        help_text.extend([
            f"{self.command_prefix}search <query> - Search for documents matching the query",
            f"{self.command_prefix}add <file_path> - Add a document to the system",
            f"{self.command_prefix}delete <id> - Delete a document from the system",
            f"{self.command_prefix}list [--limit=N] - List available documents",
            f"{self.command_prefix}settings <option> <value> - Modify system configuration",
            f"{self.command_prefix}history [--limit=N] - Show conversation history",
            f"{self.command_prefix}clear - Clear conversation history",
            f"{self.command_prefix}context [--size=N] - Manage conversation context"
        ])
        
        return "\n".join(help_text)
    
    def _get_specific_help(self, command_name: str) -> str:
        """
        Generate specific help for a command.
        
        Args:
            command_name: Command name
            
        Returns:
            Detailed help text
        """
        help_texts = {
            "help": f"""
{self.command_prefix}help [command]

Shows general help or specific help for a command.

Arguments:
  command - (Optional) Name of the command to see detailed help for
""",
            "search": f"""
{self.command_prefix}search <query> [--filter=value]

Search for documents matching the query.

Arguments:
  query - Text to search in documents

Options:
  --filter=value - Filter results by metadata (e.g.: --filter=filetype=pdf)
  --limit=N - Limit number of results (default: 5)
""",
            "add": f"""
{self.command_prefix}add <file_path> [--tags=tag1,tag2]

Add a document to the system.

Arguments:
  file_path - Path to file to add (supports .txt, .pdf, .docx, .md, .html)

Options:
  --tags=tag1,tag2 - Tags to categorize the document
""",
            "delete": f"""
{self.command_prefix}delete <id>

Delete a document from the system.

Arguments:
  id - Identifier of the document to delete
""",
            "list": f"""
{self.command_prefix}list [--limit=N] [--filter=value]

List available documents in the system.

Options:
  --limit=N - Limit number of results (default: 10)
  --filter=value - Filter results by metadata (e.g.: --filter=filetype=pdf)
""",
            "settings": f"""
{self.command_prefix}settings <option> <value>

Modify system configuration.

Arguments:
  option - Name of option to modify (temperature, top_p, etc.)
  value - New value for the option
""",
            "history": f"""
{self.command_prefix}history [--limit=N]

Show conversation history.

Options:
  --limit=N - Number of messages to show (default: all)
""",
            "clear": f"""
{self.command_prefix}clear

Clear current conversation history.
""",
            "context": f"""
{self.command_prefix}context [--size=N]

Show or modify conversation context size.

Options:
  --size=N - Set the number of messages to keep in context
"""
        }
        
        return help_texts.get(command_name, f"No help available for {command_name}")
