import sqlite3
import csv
import matplotlib.pyplot as plt
import numpy as np

DB_PATH = '/content/participants.db'


class ExperimentDB:
    def __init__(self, db_path=DB_PATH):
        try:
            self.conn = sqlite3.connect(db_path)
            print("db opened")
        except Exception as e:
            print(f"Cannot open the db: {e}")
            return

    def sql(self, query):
        try:
            self.conn.execute(query)
            self.conn.commit()
            print(f"Query executed! {query}")
        except Exception as e:
            print(f"Exception {e}")
            return

    def get_trials_data(self):
        query = """
        SELECT t.id AS trial_id, d.time, d.value, t.condition
        FROM data d
        JOIN trial t ON d.trial_id = t.id
        JOIN experiment e ON t.experiment_id = e.id
        WHERE e.completed = 1
        ORDER BY t.id, d.time;
        """
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def get_condition(self, trial_id):
        query = f"SELECT condition FROM trial WHERE id = {trial_id};"
        cursor = self.conn.execute(query)
        result = cursor.fetchone()
        return result[0] if result else None

    def get_max_time(self, condition):
        query = f"""
        SELECT MAX(time)
        FROM data d
        JOIN trial t ON d.trial_id = t.id
        WHERE t.condition = '{condition}';
        """
        cursor = self.conn.execute(query)
        result = cursor.fetchone()
        return result[0] if result else None

    def list_tables(self):
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        cursor = self.conn.execute(query)
        return [row[0] for row in cursor.fetchall()]

    def count_rows(self, table_name):
        query = f"SELECT COUNT(*) FROM {table_name};"
        cursor = self.conn.execute(query)
        result = cursor.fetchone()
        return result[0] if result else 0

    def count_valid_experiments(self):
        query = f"SELECT COUNT(*) FROM experiment WHERE completed = 1;"
        cursor = self.conn.execute(query)
        result = cursor.fetchone()
        return result[0] if result else 0


    def table_info(self, table_name):
        query = f"PRAGMA table_info({table_name});"
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def delete_items(self, table_name, column_name, values):
        if not table_name or not column_name or not values:
            print("Table name, column name, and values must be provided for deletion.")
            return

        # Validate that all items in the values list are of appropriate type
        if not all(isinstance(value, (int, str)) for value in values):
            print("Invalid values provided. All values must be integers or strings.")
            return

        placeholders = ','.join('?' for _ in values)
        query = f"DELETE FROM {table_name} WHERE {column_name} IN ({placeholders})"

        try:
            self.conn.execute(query, values)
            self.conn.commit()
            print(f"Deleted items from {table_name} where {column_name} IN ({values})")
        except Exception as e:
            print(f"Exception during deletion: {e}")

    def interpolate_trials(self, condition, num_points=100):
        # Get the trials data
        data = self.get_trials_data()
        
        # Organize data by condition
        trial_data = self.organize_by_condition(data)
        
        if condition not in trial_data:
            print(f"No data found for condition: {condition}")
            return
        
        condition_trials = trial_data[condition]

        # Determine common time points for interpolation
        min_time = min(min(time for time, _ in trial) for trial in condition_trials.values())
        max_time = max(max(time for time, _ in trial) for trial in condition_trials.values())
        common_times = np.linspace(min_time, max_time, num_points)
        
        interpolated_data = {}

        for trial_id, trial in condition_trials.items():
            times, values = zip(*trial)
            interpolator = interp1d(times, values, kind='linear', fill_value='extrapolate')
            interpolated_values = interpolator(common_times)
            interpolated_data[trial_id] = list(zip(common_times, interpolated_values))
        
        return interpolated_data
        
    def close(self):
        self.conn.close()


class DataProcessor:
    @staticmethod
    def preprocess_data(data, max_time, last_value):
        if len(data) >= 5:
            avg_first_five = np.mean([value for _, value in data[:5]])
            data[0] = (data[0][0], avg_first_five)

        data.append((max_time, last_value))

        return data


class FileManager:
    @staticmethod
    def write_csv(file_name, data):
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'value'])
            writer.writerows(data)
        print(f"Created file: {file_name}")

    @staticmethod
    def prompt_download(file_name):
        user_input = input(f"Do you want to download {file_name}? (yes/no): ")
        if user_input.lower() == 'yes':
            print(f"Downloading {file_name}...")
        else:
            print(f"Skipping download for {file_name}")


class DataPlotter:
    @staticmethod
    def plot(trial_data, condition, m_time, preprocess=True):
        plt.figure(figsize=(10, 6))

        for trial_id, data in trial_data.items():
            if preprocess:
                max_time = max([time for time, _ in data])
                last_value = data[-1][1]
                processed_data = DataProcessor.preprocess_data(data[:], max_time, last_value)
                times, values = zip(*processed_data)
                label = f'Trial {trial_id} (Processed)'
            else:
                times, values = zip(*data)
                label = f'Trial {trial_id}'

            plt.plot(times, values, label=label)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Time/Value Pairs for Condition: {condition}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside to the right
        plt.grid(True)
        plt.show()
        plt.close()  # Ensure the current figure is closed to release resources


class ExperimentProcessor:
    def __init__(self):
        self.db = ExperimentDB()
        self.file_mgr = FileManager()

    def process_and_plot(self, desired_conditions = [], desired_trials = [], preprocess=True):
        data = self.db.get_trials_data()
        trial_data = self.organize_by_condition(data)
        for condition, trials in trial_data.items():
            if len(desired_conditions) == 0:
                self.plot_data(trials, condition, preprocess)
            if condition in desired_conditions:
                self.plot_data(trials, condition, preprocess)

        self.db.close()

    def plot_trials(self, desired_trials=[], preprocess=False):
        data = self.db.get_trials_data()
        trial_data = self.organize_by_condition(data)

        for condition, trials in trial_data.items():
            for trial_id in desired_trials:
                if trial_id in trials:
                    trial_data_to_plot = {trial_id: trials[trial_id]}
                    self.plot_data(trial_data_to_plot, condition)

        self.db.close()

    def get_trials_per_condition(self, cond):
        data = self.db.get_trials_data()
        trial_data = self.organize_by_condition(data)
        return trial_data[cond]

    def get_trial(self, desired_trial=[]):
        data = self.db.get_trials_data()
        trial_data = self.organize_by_condition(data)

        for condition, trials in trial_data.items():
            for trial_id in desired_trial:
                if trial_id in trials:
                    return trials[trial_id]
        self.db.close()

    def organize_by_condition(self, data):
        trial_data = {}
        for row in data:
            trial_id, time, value, condition = row
            if condition not in trial_data:
                trial_data[condition] = {}
            if trial_id not in trial_data[condition]:
                trial_data[condition][trial_id] = []
            trial_data[condition][trial_id].append((time, value))

        return trial_data

    def plot_data(self, trial_data, condition, preprocess=True):
        m_time = self.db.get_max_time(condition)
        DataPlotter.plot(trial_data, condition, m_time, preprocess)

    def save_to_csv(self, trial_data, condition):
        for trial_id, data in trial_data.items():
            file_name = f"{trial_id}_{condition}.csv"
            self.file_mgr.write_csv(file_name, data)
            self.file_mgr.prompt_download(file_name)
