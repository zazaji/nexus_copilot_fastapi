# backend/app/services/knowledge_graph_service.py
import sqlite3
import re
import logging
import time
import os
from typing import List, Set, Dict, Any
from pydantic import BaseModel

class NotePayload(BaseModel):
    file_path: str
    content: str
    title: str

def find_wikilinks(content: str) -> Set[str]:
    """Extracts unique [[WikiLink]] targets from text content with robust Unicode support."""
    # Use a more permissive regex to capture anything between the brackets.
    pattern = re.compile(r'\[\[([^\]]+)\]\]')
    matches = pattern.findall(content)
    # Log Point 4: Check regex matching results
    logging.info(f"[DEBUG_WIKILINK] find_wikilinks content (first 50 chars): '{content[:50]}...', matches: {matches}")
    return set(match.strip() for match in matches if match.strip())

def find_or_create_note_path_by_title(conn: sqlite3.Connection, title: str) -> str | None:
    """
    Finds the file path (ID) of a note by its title, case-insensitively.
    If not found, creates a 'ghost' note entry and returns its ID.
    """
    cursor = conn.cursor()
    # Use COLLATE NOCASE for case-insensitive title matching.
    cursor.execute("SELECT id FROM notes WHERE title = ? COLLATE NOCASE", (title,))
    result = cursor.fetchone()
    if result:
        return result[0]
    
    # Fallback to check file path for real notes, also case-insensitively on the title part.
    # This is less efficient but robust. A better schema might store a normalized title.
    cursor.execute("SELECT id, file_path FROM notes")
    all_notes = cursor.fetchall()
    for row in all_notes:
        note_id, file_path = row
        if not file_path.startswith("ghost://"):
            file_title = os.path.splitext(os.path.basename(file_path))[0]
            if file_title.lower() == title.lower():
                return note_id

    logging.info(f"Creating ghost note for non-existent link: '{title}'")
    try:
        ghost_id = f"ghost::{title}"
        ghost_path = f"ghost://{title}.md"
        current_time = int(time.time() * 1000)
        cursor.execute(
            "INSERT OR IGNORE INTO notes (id, file_path, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (ghost_id, ghost_path, title, current_time, current_time)
        )
        return ghost_id
    except Exception as e:
        logging.error(f"Failed to create ghost note for '{title}': {e}")
        return None


def update_links_for_note(conn: sqlite3.Connection, note_path: str, content: str):
    """
    Parses a note's content for links and updates the database.
    """
    logging.info(f"Updating links for note: {note_path}")
    source_id = note_path
    linked_titles = find_wikilinks(content)

    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        target_ids = set()
        for title in linked_titles:
            target_path = find_or_create_note_path_by_title(conn, title)
            if target_path and target_path != source_id:
                target_ids.add(target_path)
            else:
                logging.warning(f"Could not find or create note with title '{title}' linked from '{note_path}'")

        cursor.execute("DELETE FROM note_links WHERE source_id = ?", (source_id,))
        if target_ids:
            links_to_insert = [(source_id, target_id) for target_id in target_ids]
            cursor.executemany("INSERT OR IGNORE INTO note_links (source_id, target_id) VALUES (?, ?)", links_to_insert)

        conn.commit()
        logging.info(f"Successfully updated {len(target_ids)} links for note '{note_path}'.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Database transaction failed while updating links for '{note_path}': {e}", exc_info=True)
        raise

def rebuild_all_links(conn: sqlite3.Connection, notes: List[NotePayload]):
    """
    Atomically clears and rebuilds the entire knowledge graph from a list of all notes.
    """
    logging.info(f"Starting atomic full rebuild of knowledge graph with {len(notes)} notes.")
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        cursor.execute("DELETE FROM note_links")
        cursor.execute("DELETE FROM notes")
        logging.info("Cleared existing notes and links.")

        current_time = int(time.time() * 1000)
        notes_to_insert = [
            (note.file_path, note.file_path, note.title, current_time, current_time)
            for note in notes
        ]
        if notes_to_insert:
            cursor.executemany(
                "INSERT INTO notes (id, file_path, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                notes_to_insert
            )
            logging.info(f"Bulk inserted {len(notes_to_insert)} real notes.")

        # Build a case-insensitive map for robust matching
        title_to_path_map = {note.title.lower(): note.file_path for note in notes}
        all_linked_titles = set()
        for note in notes:
            filename = os.path.basename(note.file_path)
            if filename.endswith('.md'):
                title_to_path_map[filename[:-3].lower()] = note.file_path
            all_linked_titles.update(find_wikilinks(note.content))

        ghosts_to_create = []
        for title in all_linked_titles:
            if title.lower() not in title_to_path_map:
                ghost_id = f"ghost::{title}"
                ghost_path = f"ghost://{title}.md"
                ghosts_to_create.append(
                    (ghost_id, ghost_path, title, current_time, current_time)
                )
        
        if ghosts_to_create:
            cursor.executemany(
                "INSERT OR IGNORE INTO notes (id, file_path, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ghosts_to_create
            )
            logging.info(f"Created {len(ghosts_to_create)} ghost notes.")

        all_notes_in_db = cursor.execute("SELECT id, title, file_path FROM notes").fetchall()
        # Build final map case-insensitively
        final_title_to_id_map = {row['title'].lower(): row['id'] for row in all_notes_in_db}
        for row in all_notes_in_db:
            filename = os.path.basename(row['file_path'])
            if filename.endswith('.md'):
                final_title_to_id_map[filename[:-3].lower()] = row['id']

        all_links_to_insert = []
        for note in notes:
            source_id = note.file_path
            linked_titles = find_wikilinks(note.content)
            for title in linked_titles:
                target_id = final_title_to_id_map.get(title.lower())
                if target_id and target_id != source_id:
                    all_links_to_insert.append((source_id, target_id))

        if all_links_to_insert:
            cursor.executemany("INSERT OR IGNORE INTO note_links (source_id, target_id) VALUES (?, ?)", all_links_to_insert)
            logging.info(f"Bulk inserted {len(all_links_to_insert)} links.")

        conn.commit()
        logging.info("Successfully and atomically rebuilt knowledge graph.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Database transaction failed during full graph rebuild: {e}", exc_info=True)
        raise

def get_graph_data(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Fetches all nodes and links for the graph visualization."""
    conn.row_factory = sqlite3.Row
    nodes_query = "SELECT id, title, file_path FROM notes"
    links_query = "SELECT source_id, target_id FROM note_links"
    
    nodes = [
        {
            "id": row["id"],
            "label": row["title"],
            "type": "ghost" if row["id"].startswith("ghost::") else "real"
        }
        for row in conn.execute(nodes_query).fetchall()
    ]
    links = [{"source": row["source_id"], "target": row["target_id"]} for row in conn.execute(links_query).fetchall()]
    
    return {"nodes": nodes, "links": links}

def get_note_details(conn: sqlite3.Connection, note_id: str) -> Dict[str, Any] | None:
    """Fetches details for a single note, including its links."""
    conn.row_factory = sqlite3.Row
    note_query = "SELECT id, file_path, title, created_at, updated_at FROM notes WHERE id = ?"
    backlinks_query = "SELECT n.id, n.title FROM note_links l JOIN notes n ON l.source_id = n.id WHERE l.target_id = ?"
    outgoing_links_query = "SELECT n.id, n.title FROM note_links l JOIN notes n ON l.target_id = n.id WHERE l.source_id = ?"

    note_row = conn.execute(note_query, (note_id,)).fetchone()
    if not note_row:
        return None

    content = ""
    file_path = note_row["file_path"]
    if not file_path.startswith("ghost://") and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Could not read content for note {note_id} from path {file_path}: {e}")
            content = f"Error: Could not read file content from {file_path}."

    backlinks = [{"note_id": row["id"], "note_title": row["title"]} for row in conn.execute(backlinks_query, (note_id,)).fetchall()]
    outgoing_links = [{"note_id": row["id"], "note_title": row["title"]} for row in conn.execute(outgoing_links_query, (note_id,)).fetchall()]

    return {
        "id": note_row["id"],
        "file_path": file_path,
        "title": note_row["title"],
        "content": content,
        "created_at": note_row["created_at"],
        "updated_at": note_row["updated_at"],
        "backlinks": backlinks,
        "outgoing_links": outgoing_links,
    }