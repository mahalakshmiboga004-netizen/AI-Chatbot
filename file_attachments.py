"""Validation and preparation of text-file attachments for chat messages."""


MAX_FILE_SIZE = 1024 * 1024
MAX_FILES_PER_MESSAGE = 5
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".log",
    ".sql",
    ".sh",
    ".ps1",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".go",
    ".rs",
}


def prepare_attachments(attachments):
    """Return validated attachment text for the model prompt."""
    if not isinstance(attachments, list):
        raise ValueError("Attachments must be a list.")
    if len(attachments) > MAX_FILES_PER_MESSAGE:
        raise ValueError(f"Attach no more than {MAX_FILES_PER_MESSAGE} files.")

    contents = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ValueError("Invalid file attachment.")

        supplied_name = attachment.get("name")
        content = attachment.get("content")
        if not isinstance(supplied_name, str) or not isinstance(content, str):
            raise ValueError("Each attachment must include a file name and text.")

        name = supplied_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if (
            not name
            or len(name) > 120
            or any(ord(character) < 32 for character in name)
            or "." + name.rsplit(".", 1)[-1].lower()
            not in SUPPORTED_EXTENSIONS
        ):
            raise ValueError(f"Unsupported text file: {name or 'unnamed file'}.")

        try:
            size = len(content.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError(f"{name} is not valid UTF-8 text.") from error
        if size > MAX_FILE_SIZE:
            raise ValueError(f"{name} exceeds the 1 MiB file limit.")
        if "\0" in content:
            raise ValueError(f"{name} contains binary data, not plain text.")

        contents.append(f"[Attached file: {name}]\n{content}")

    return "\n\n".join(contents)
