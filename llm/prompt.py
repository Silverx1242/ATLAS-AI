class PromptManager:
    """
    Class to handle prompts flexibly.
    """

    def __init__(self):
        # Dictionary to store prompt templates
        self.prompts = {}

    def add_prompt(self, name: str, template: str):
        """
        Add a new prompt template.

        :param name: Prompt name.
        :param template: Prompt template.
        """
        self.prompts[name] = template

    def get_prompt(self, name: str, **kwargs) -> str:
        """
        Get a formatted prompt with provided values.

        :param name: Prompt name.
        :param kwargs: Values to format the template.
        :return: Formatted prompt.
        """
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found.")
        return self.prompts[name].format(**kwargs)

    def list_prompts(self) -> list:
        """
        List all available prompt names.

        :return: List of prompt names.
        """
        return list(self.prompts.keys())
