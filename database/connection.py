from sqlalchemy import create_engine
import json
import logging
from typing import Dict
import os
from core.config import Config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    def __init__(self):
        self.connections_file = Config.DATABASE_CONNECTIONS_FILE
        self._migrate_connections_if_needed()

    def _migrate_connections_if_needed(self):
        """Migre les anciennes connexions vers le nouveau format."""
        try:
            if os.path.exists(self.connections_file):
                with open(self.connections_file, 'r') as f:
                    connections = json.load(f)

                needs_migration = False
                migrated_connections = {}

                for name, details in connections.items():
                    if 'name' not in details:
                        needs_migration = True
                        migrated_connections[name] = {
                            'name': name,
                            'db_type': details.get('db_type', ''),
                            'user': details.get('user', ''),
                            'password': details.get('password', ''),
                            'host': details.get('host', ''),
                            'port': details.get('port', ''),
                            'database': details.get('database', '')
                        }
                    else:
                        migrated_connections[name] = details

                if needs_migration:
                    with open(self.connections_file, 'w') as f:
                        json.dump(migrated_connections, f, indent=2)
                    logger.info("Connections migrated to new format")

        except Exception as e:
            logger.error(f"Error during connection migration: {str(e)}")

    def get_db_engine(self, db_type: str, user: str, password: str, host: str, port: str, database: str):
        connection_strings = {
            "mysql": f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}",
            "postgresql": f"postgresql://{user}:{password}@{host}:{port}/{database}",
            "mssql": f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
        }

        if db_type not in connection_strings:
            raise ValueError("Unsupported database type")

        return create_engine(connection_strings[db_type])

    def test_connection(self, engine) -> bool:
        try:
            with engine.connect() as conn:
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def load_connections(self) -> Dict:
        try:
            if os.path.exists(self.connections_file):
                with open(self.connections_file, 'r') as f:
                    connections = json.load(f)
                return connections
            return {}
        except Exception as e:
            logger.error(f"Error loading connections: {str(e)}")
            return {}

    def save_connection(self, name: str, details: Dict):
        """
        Sauvegarde une connexion avec une structure cohérente.

        Args:
            name: Nom de la connexion
            details: Détails de la connexion
        """
        connections = self.load_connections()

        connection_data = {
            'name': name,
            'db_type': details.get('db_type', ''),
            'user': details.get('user', ''),
            'password': details.get('password', ''),
            'host': details.get('host', ''),
            'port': details.get('port', ''),
            'database': details.get('database', '')
        }

        connections[name] = connection_data

        try:
            with open(self.connections_file, 'w') as f:
                json.dump(connections, f, indent=2)
            logger.info(f"Connection {name} saved successfully")
        except Exception as e:
            logger.error(f"Error saving connection: {str(e)}")
            raise