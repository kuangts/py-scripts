import sqlite3
import os
import dicom
from datetime import datetime
from random import randint
import shutil
from contextlib import contextmanager, closing


@contextmanager
def connect(location):
    cm = case_manager(location)
    try:
        yield cm.__enter__()
    finally:
        cm.__exit__()


def new(location, columns={}):
    if not columns:
        columns = {
        "Study_ID": "VARCHAR(64) NOT NULL",
        "Subject_ID": "VARCHAR(64) NOT NULL",
        "Series_ID": "VARCHAR(64) NOT NULL PRIMARY KEY",
        "Patient_Name": "VARCHAR",
        "Patient_Gender": "VARCHAR",
        "Patient_Age": "INTEGER",
        "Patient_Birth_Date": "DATE",
        "Instance_Creation_Date": "DATE",
        "Study_Date": "DATE",
        "Series_Date": "DATE",
        "Patient_Comments": "TEXT",
        }
    if isinstance(columns, list):
        columns = dict(zip(columns,["VARCHAR"]*len(columns)))
    with connect(location) as cm:
        with closing(cm.cursor) as cur:
            cur.execute(" CREATE TABLE IF NOT EXISTS DICOM ( {} ); ".format(
                ','.join(
                    [ ' '.join(z) for z in columns.items()] + ["Entry_Date DATE"]
                )
            ))

class case_manager:

    def __repr__(self):
        with closing(self.cursor) as cur:
            for row in cur.fetchall():
                print(row)

    def __enter__(self):
        print('__enter__ called')
        self._connection = sqlite3.connect(self.location)
        return self

    def __exit__(self, *args, **kwargs):
        print('__exit__ called')
        self._connection.commit()
        self._connection.__exit__(*args, **kwargs)
        self._connection = None

    @property
    def connection(self):
        if hasattr(self, '_connection') and self._connection is not None:
            return self._connection
        else:
            self.__enter__()
            return self._connection

    @property
    def cursor(self):
        return self.connection.cursor()

    @classmethod
    def default(cls):
        return cls(os.path.join(os.path.dirname(__file__), 'asset', 'default.db'))

    def __init__(self, location='', columns={}):
        if not location:
            location = os.path.join(os.getcwd(), 'default.db')
        location = os.path.realpath(os.path.expanduser(location))
        if not os.path.exists(location):
            new(location, columns=columns)
            pass
        setattr(self, 'location', location)
        setattr(self, '_connection', None)

    def columns(self):
        with closing(self.cursor) as cur:
            return [i[0] for i in cur.execute('''SELECT * FROM Dicom''').description]

    def insert(self, files):
        info = dicom.read(files, return_info=True, return_image=False)

        newrow = ()
        sql = "INSERT INTO Dicom("
        for key in self.tags:
            sql = sql + key + ','
            if key is "Subject_ID" or key is "Series_ID":
                if info.get(self.tags.get(key)) is None:
                    newrow = newrow + (str(randint(100000000000, 999999999999)),)
                else:
                    newrow = newrow + (info.get(self.tags.get(key)),)
            elif key.find("Date") != -1 and info.get(self.tags.get(key)) is not None:
                s = info.get(self.tags.get(key))
                newrow = newrow + (s[:4] + "-" + s[4:6] + "-" + s[6:],)
            else:
                newrow = newrow + (info.get(self.tags.get(key)),)
        newrow = newrow + (datetime.today().strftime('%Y-%m-%d'),)
        sql = sql + "Entry_Date) "
        sql = sql + "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
        with closing(self.cursor) as cur:
            cur.execute(sql, newrow)

    def read(self, **tags):
        sql = "SELECT * FROM Dicom WHERE "
        for key, value in tags.items():
            sql = sql + f' {key} = {value} and'
        sql = sql[:-4]
        sql = sql.rstrip(" WHERE")
        with closing(self.cursor) as cur:
            cur.execute(sql)

    def readanywhere(self, tag):
        sql = f"SELECT * FROM Dicom WHERE \"{tag}\" IN ("
        for key in self.columns():
            sql = sql + key + ','
        sql = sql.rstrip(",") + ")"
        with closing(self.cursor) as cur:
            cur.execute(sql)

    def readall(self):
        with closing(self.cursor) as cur:
            cur.execute("SELECT * FROM Dicom")

    def update(self, update={}, where={}):
        sql = "UPDATE Dicom SET "
        for key in update:
            sql = sql + f" {key} = \"{update.get(key)}\","
        sql = sql.rstrip(",") + " WHERE "
        for key, value in where.items():
            sql = sql + key + "=" + value + " and "
        sql = sql[:-4]
        with closing(self.cursor) as cur:
            cur.execute(sql)

    def delete(self, **tags):
        sql = 'DELETE FROM Dicom WHERE '
        for key, value in tags.items():
            sql = sql + f" {key} = \'{value}\' and"
        sql = sql[:-3]
        with closing(self.cursor) as cur:
            cur.execute(sql)

    # def fuzzydelete(self, tag):
    #     sql = f'DELETE FROM Dicom WHERE \"{tag}\" IN ('
    #     for key in self.tags:
    #         sql = sql + key + ','
    #     sql = sql.rstrip(",") + ')'
    #     with closing(self.cursor) as cur:
    #         cur.execute(sql)

if __name__ == "__main__":
    with case_manager.default() as cm:
        # cm.readall()
        # cm.update({"Case_Name": "case2"}, {"Series_ID": "663201465469"})
        # cm.readall()
        cm.readanywhere("case1")

