"""Character repository for database access.

This module provides database access for character information.
CharacterRepository fetches character data from MySQL database with automatic retry logic.
"""
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


class ICharacterRepository(ABC):
    """Character database access interface.

    Defines the contract for retrieving character data from storage.
    Implementations handle the actual database interaction logic.
    """

    @abstractmethod
    def get_characters_by_group(self, group_name: str, workflow_name: str) -> List[str]:
        """Fetch character list for a group.

        Retrieves all active characters belonging to a specific group that support
        a given workflow type.

        Args:
            group_name: Character group identifier (e.g., 'Kirby', 'Pokemon')
            workflow_name: Workflow type identifier (currently unused in query)

        Returns:
            List of character name strings (English names)
        """
        pass

    @abstractmethod
    def get_random_character_from_group(self, group_name: str, workflow_name: str) -> Optional[str]:
        """Select random character from a group.

        Retrieves all characters in the group, then randomly selects one.
        Useful for variety in multi-character scenarios.

        Args:
            group_name: Character group identifier
            workflow_name: Workflow type identifier

        Returns:
            Single character name string, or None if group is empty
        """
        pass


class CharacterRepository(ICharacterRepository):
    """MySQL-based character database access.

    Implements character retrieval from MySQL database with automatic
    connection retry on failure. Queries the anime.anime_roles table.

    Database schema expected:
        anime.anime_roles:
            - role_name_en: English character name
            - group_name: Character group identifier
            - status: Active flag (1 = active)
            - weight: Selection weight (>0 for selectable characters)

    Args:
        db_connection: Database connection object with cursor attribute
    """

    def __init__(self, db_connection):
        """Initialize repository with database connection.

        Args:
            db_connection: Database connection object
        """
        self.db_connection = db_connection

    def get_characters_by_group(self, group_name: str, workflow_name: str) -> List[str]:
        """Fetch character list for a group with retry.

        Queries database for active characters in the specified group.
        Automatically retries up to 3 times on database errors, reconnecting
        between attempts.

        Args:
            group_name: Character group identifier
            workflow_name: Workflow type (currently unused in query)

        Returns:
            List of character English names

        Raises:
            Exception: Database error after 3 retry attempts
        """
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                cursor = self.db_connection.cursor
                query = """
                    SELECT role_name_en
                    FROM anime.anime_roles
                    WHERE group_name = %s AND status = 1 AND weight > 0
                """.strip()

                cursor.execute(query, (group_name,))
                results = cursor.fetchall()
                return [row[0] for row in results]

            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise

                # Reconnect to database and retry
                from lib.database import db_pool
                self.db_connection = db_pool.get_connection('mysql')
                continue

    def get_random_character_from_group(self, group_name: str, workflow_name: str) -> Optional[str]:
        """Select random character from a group.

        Fetches all characters in the group using get_characters_by_group(),
        then randomly selects one.

        Args:
            group_name: Character group identifier
            workflow_name: Workflow type identifier

        Returns:
            Random character English name, or None if group has no characters
        """
        import random
        characters = self.get_characters_by_group(group_name, workflow_name)
        if not characters:
            return None
        return random.choice(characters)
