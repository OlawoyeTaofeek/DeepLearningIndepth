import os
import re
from zipfile import ZipFile
import pandas as pd

def read_whatsapp_chat(data_dir: str = "data") -> pd.DataFrame:
    """
    Parse one or more WhatsApp chat exports into a structured DataFrame.

    This function scans a directory for WhatsApp `.zip` export files, extracts
    them, reads the contained `.txt` chat logs, filters out noise (system
    messages, media placeholders, deleted messages, URLs, emails, etc.),
    normalises platform-specific formatting differences between iOS and Android
    exports, and parses each message into its constituent parts.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing one or more WhatsApp `.zip` export
        files. Defaults to ``"data"``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:

        - ``timestamp`` (datetime64[ns]): When the message was sent.
        - ``sender`` (str): Display name or phone number of the sender.
        - ``message`` (str): Cleaned message body.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist or contains no ``.zip`` files.
    StopIteration
        If an extracted zip archive contains no ``.txt`` file.

    Notes
    -----
    - Supports both **iOS** (square-bracket timestamps, narrow no-break spaces)
      and **Android** (dash-separated, 12/24-hour clock) export formats.
    - Unicode artefacts such as Left-to-Right Mark (U+200E) and Right-to-Left
      Mark (U+200F) are stripped automatically.
    - Each zip is extracted into its own sub-folder named after the zip file to
      prevent filename collisions across multiple exports.
    - System events (e.g. "X added Y", "You pinned a message") are detected via
      both exact string matching and a dedicated regex and excluded from output.

    Examples
    --------
    >>> df = read_whatsapp_chat("data")
    Found 2 zip files: ['chat1.zip', 'chat2.zip']
      ✓ chat1.zip: 142,300 characters loaded
      ✓ chat2.zip: 98,450 characters loaded
    Total characters across all chats: 240,750
    Total messages parsed: 3,847

    >>> df.head()
              timestamp          sender                 message
    0 2023-01-01 08:45    Temitayo G.   Good morning everyone!
    1 2023-01-01 08:46  +234 801 000 1  Morning! How are you?
    """

    FILTER_STRINGS = (
        "end-to-end encrypted", 
        "tap to learn more",
        "media omitted",
        "you deleted this message",
        "this message was deleted",
        "created group",
        "added you",
        "you pinned a message",
        "changed their phone number",
        "joined using this group's invite link",
        "changed the subject from",
        "changed this group's icon",
        "changed the group description",
        "turned on disappearing messages",
        "turned off disappearing messages",
        "security code with",
        "your security code",
        "missed voice call",
        "missed video call",
    )

    SYSTEM_EVENT_PATTERN = re.compile(
        r'^'
        r'(?:\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?\s*[-~]\s*)?'
        r'(?:[^:]+?)\s+'
        r'(?:added|pinned|removed|left|joined|changed|created|was added|were added)'
        r'(?:\s|$)',
        re.IGNORECASE,
    )

    EMAIL_PATTERN   = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
    URL_PATTERN     = re.compile(r'https?://\S+')
    TAGGING_PATTERN = re.compile(r'@\w+')
    EDITED_MARKER   = "<This message was edited>"
    NULL_SUFFIX     = "null"

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: '{data_dir}'")

    zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in '{data_dir}'")

    print(f"Found {len(zip_files)} zip file(s): {zip_files}\n")

    all_text = ""

    for zip_file in zip_files:
        zip_path   = os.path.join(data_dir, zip_file)
        extract_to = os.path.join(data_dir, zip_file.removesuffix(".zip"))

        with ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)

        txt_file = next(
            os.path.join(extract_to, f)
            for f in os.listdir(extract_to)
            if f.endswith(".txt")
        )

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        all_text += text + "\n"
        print(f"  ✓ {zip_file}: {len(text):,} characters loaded")

    print(f"\nTotal characters across all chats: {len(all_text):,}")

    all_text = all_text.replace("\u202f", " ")
    all_text = all_text.replace("\u200E", "").replace("\u200F", "")

    filtered_lines = []

    for line in all_text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue
        # Lowercase comparison ensures phrases match regardless of platform casing
        if any(phrase in stripped.lower() for phrase in FILTER_STRINGS):
            continue
        if SYSTEM_EVENT_PATTERN.search(stripped):
            continue
        if stripped.split()[-1] == NULL_SUFFIX:
            continue
        if EMAIL_PATTERN.search(stripped) or URL_PATTERN.search(stripped):
            continue

        stripped = stripped.replace(EDITED_MARKER, "").strip()
        stripped = TAGGING_PATTERN.sub("", stripped).strip()

        if stripped:
            filtered_lines.append(stripped)

    content = "\n".join(filtered_lines)
    content = re.sub(
        r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm])\]',
        r'\1',
        content,
    )

    MSG_PATTERN = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)'
        r'\s*[-~]\s*'
        r'([^:]+?)'
        r':\s*'
        r'(.*?)'
        r'(?=\n\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}|$)',
        re.DOTALL,
    )

    messages = MSG_PATTERN.findall(content)
    df = pd.DataFrame(messages, columns=["timestamp", "sender", "message"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df["sender"]    = df["sender"].str.strip()
    df["message"]   = df["message"].str.strip()

    # Drop rows where sender still looks like a system event
    SYSTEM_SENDER_PATTERN = re.compile(
        r'\b(added|pinned|removed|left|joined|changed|created|encrypted)\b',
        re.IGNORECASE,
    )
    df = df[~df["sender"].str.contains(SYSTEM_SENDER_PATTERN, regex=True)].reset_index(drop=True)

    VALID_SENDER_PATTERN = re.compile(
        r'^(\+?[\d][\d\s]{6,}|[A-Za-z][A-Za-z0-9\s._\-]{1,50})$'
    )
    df = df[df["sender"].str.strip().apply(
        lambda s: bool(VALID_SENDER_PATTERN.match(s))
    )].reset_index(drop=True)

    df["message"] = df["message"].str.replace(r'\d+', '', regex=True).str.strip()

    # Clean up any leftover punctuation artifacts from removing numbers (e.g., empty hyphens)
    df["message"] = df["message"].str.replace(r'^\s*[-:]\s*', '', regex=True).str.strip()

    # Drop rows with empty messages after cleaning
    df = df[df["message"].str.len() > 0].reset_index(drop=True)

    print(f"Total messages parsed: {len(df):,}")
    return df