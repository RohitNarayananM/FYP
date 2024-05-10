import sqlite3
import csv


w = csv.writer(open("data_new.csv", "w"))
w.writerow(["file_change_id", "name", "before_code", "after_code", "cweid"])

ids = open("data.csv").readlines()
ids = [i.strip() for i in ids]


def get_data():
    conn = sqlite3.connect("CVEfixes.db")
    c = conn.cursor()
    i = 0
    for l in ids:
        print(i, end="\r")
        k = c.execute(
            f"WITH changes AS (SELECT * from method_change WHERE file_change_id = '{l.strip()}') SELECT mc1.file_change_id, mc1.name, mc1.code, mc2.code FROM changes mc1 JOIN changes mc2 ON mc1.name = mc2.name AND mc1.before_change = 'True' AND mc2.before_change = 'False';"
        ).fetchall()
        for j in k:
            w.writerow(j)
            i += 1


get_data()
