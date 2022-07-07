import pickle
import sqlite3
import logging

class ResultLogger:
    def __init__(self, folder="logs", experiment="test", db_uri: str = '/l/users/roberto.guillen/nilm/logs/', 
    db_name: str = 'nilm'):
        self.folder=  folder
        self.experiment = experiment 
        self.sqlite = f'{db_uri}/{db_name}.db'

    def create_db(self):
        # check exist database 
        try:
            sqlite_connection = sqlite3.connect(self.sqlite)
            cursor = sqlite_connection.cursor()
            query = 'CREATE TABLE IF NOT EXISTS results (llave VARCHAR(255) PRIMARY KEY, result_data LONGBLOB);'
            cursor.execute(query)
            sqlite_connection.commit()
            sqlite_connection.close()
        except sqlite3.Error as error:
            print("Failed to start db ", error)
            sqlite_connection.rollback()
            sqlite_connection.close()

    def insert_result(self, key):
        try:
            # Configure connection object
            sqlite_connection = sqlite3.connect(self.sqlite)
            cursor = sqlite_connection.cursor()
            print("Connected to SQLite")
            # save the image in the databases
            query = '''INSERT INTO results (llave) VALUES (?)'''
            cursor.execute(query, (key,))
            # Handle connection close
            sqlite_connection.commit()
            sqlite_connection.close()
            print("Experiment record created")
        except sqlite3.Error as error:
            print("Failed to create experiment record", error)
            sqlite_connection.rollback()
            sqlite_connection.close()
            
    def update_result(self, key, value):
        try:
            # Configure connection object
            sqlite_connection = sqlite3.connect(self.sqlite)
            cur = sqlite_connection.cursor()
            # Update the current form measurements
            query = '''UPDATE results SET result_data = ? where llave = ? '''
            cur.execute(query, (pickle.dumps(value),key))
            # Handle connection close
            sqlite_connection.commit()
            sqlite_connection.close()
        except sqlite3.Error:
            logging.exception("Error running Query")
            sqlite_connection.rollback()
            sqlite_connection.close()
 
    def get_results(self):
        try:
            # Configure connection object
            sqlite_connection = sqlite3.connect(self.sqlite)
            cur = sqlite_connection.cursor()
            # Update the current form measurements
            query = '''SELECT * FROM results'''
            cur.execute(query)
            results = cur.fetchall()
            # Handle connection close
            sqlite_connection.commit()
            sqlite_connection.close()
            return results
        except sqlite3.Error:
            logging.exception("Error running Query")
            sqlite_connection.rollback()
            sqlite_connection.close()

    def get_result_from_key(self, key):
        try:
            # Configure connection object
            sqlite_connection = sqlite3.connect(self.sqlite)
            cur = sqlite_connection.cursor()
            # Update the current form measurements
            query = '''SELECT * FROM results WHERE llave = ?'''
            cur.execute(query, (key,))
            results = cur.fetchone()
            # Handle connection close
            sqlite_connection.commit()
            sqlite_connection.close()
            if results[1] is None: # key exists, but empty 
                return []
            return results[0],pickle.loads(results[1])
        except sqlite3.Error:
            logging.exception("Error running Query")
            sqlite_connection.rollback()
            sqlite_connection.close()
        except TypeError: # Key does not exist
            return None

    def check_if_exists(self, key):
        try:
            # Configure connection object
            sqlite_connection = sqlite3.connect(self.sqlite)
            cur = sqlite_connection.cursor()
            # Update the current form measurements
            query = '''SELECT * FROM results WHERE llave = ?'''
            cur.execute(query, (key,))
            results = cur.fetchone()
            # Handle connection close
            sqlite_connection.commit()
            sqlite_connection.close()
            return results[0]
        except sqlite3.Error:
            logging.exception("Error running Query")
            sqlite_connection.rollback()
            sqlite_connection.close()
        except TypeError: # Key does not exist
            return None
